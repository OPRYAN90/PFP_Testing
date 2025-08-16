from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional FlashAttention toggle – does nothing on unsupported systems
# -----------------------------------------------------------------------------

# from utils.flash_control import maybe_enable_flash_attention  
# maybe_enable_flash_attention(True)

# swap ESM-C SDK for ESM++ (HF-style, grad-friendly)
from transformers import AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType

from lightning import LightningModule
from torchmetrics import MeanMetric
from data.go_utils import (
    propagate_go_preds, propagate_ec_preds,
    function_centric_aupr, cafa_fmax, smin
)

# Simple helper to get number of classes without datamodule dependency
def get_num_classes_for_task(task_type: str) -> int:
    """Get number of classes for a task type."""
    class_counts = {"mf": 489, "bp": 1943, "cc": 320}
    return class_counts[task_type]

# -----------------------------------------------------------------------------
# LoRA verification utilities
# -----------------------------------------------------------------------------
def count_hits(model: nn.Module, needle: str) -> int:
    return sum(1 for name, _ in model.named_modules() if needle in name)

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w12 = nn.Linear(d_in, 2 * d_hidden)  # produces [a, b]
        self.proj = nn.Linear(d_hidden, d_in)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.proj(F.silu(a) * b)


class ResidualFeedForward(nn.Module):
    """Transformer-style FFN with SwiGLU."""
    def __init__(self, d_model: int, expansion: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # match params of GELU(4x) with SwiGLU(~8/3 x)
        d_hidden = int(round((8/3) * d_model))
        self.ff = nn.Sequential(
            SwiGLU(d_model, d_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class MLPHead(nn.Module):
    """Final classification head with layer norm and dropout."""
    
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1): 
        super().__init__()
        assert d_in == 768, "d_in must be 768"
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 602),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(602, d_out)
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean_pooling(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Simple masked mean pooling.
    
    Args:
        x: Tensor of shape [B, L, D]
        pad_mask: Tensor of shape [B, L] - True where padded
        
    Returns:
        Tensor of shape [B, D] - mean pooled representation
    """
    # Zero out padded positions
    x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    
    # Compute lengths (number of non-padded positions)
    lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp_min(1)
    
    # Sum and divide by lengths
    pooled = x_masked.sum(dim=1) / lengths
    
    return pooled
    
class AttentionFusion(nn.Module):
    def __init__(self, d_model=768, p_drop=0.1, favor_seq_bias=0.0):
        super().__init__()
        self.pre = nn.LayerNorm(2 * d_model)
        self.dropout = nn.Dropout(p_drop)
        self.gate = nn.Linear(2 * d_model, 2)
        # Optional: bias > 0 favors seq at init; 0.0 is neutral (50/50)
        nn.init.constant_(self.gate.bias, favor_seq_bias)

        # (Nice default) zero-init weights so bias controls the initial mix
        nn.init.zeros_(self.gate.weight)

    def forward(self, left_feat, right_feat):
        h = torch.cat([left_feat, right_feat], dim=-1)
        logits = self.gate(self.dropout(self.pre(h)))   # [B, 2]
        w = torch.softmax(logits, dim=-1)               # [B, 2]
        return w[..., 0:1] * left_feat + w[..., 1:2] * right_feat

############################################################
#  Encoders for modality pairs
############################################################

class CrossAttentionFusionTwoModalities(nn.Module):
    """Cross-attention between two modalities with softmax gating"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention for each modality pair
        self.mod1_to_mod2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mod2_to_mod1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Per-token gating over {mod1, mod2}
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)  # logits -> softmax outside
        )
        
        # Final combination layer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, mod1, mod2, pad_mask=None):
        # Apply padding mask to query tensors before cross-attention
        if pad_mask is not None:
            # Zero out padded positions in query tensors
            mod1 = mod1.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            mod2 = mod2.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        
        # Cross-attention with correct padding masks
        mod1_attended, _ = self.mod1_to_mod2(mod1, mod2, mod2, key_padding_mask=pad_mask)
        mod2_attended, _ = self.mod2_to_mod1(mod2, mod1, mod1, key_padding_mask=pad_mask)
        
        # Convex (softmax) gating per token over the two streams
        gate_in = torch.cat([mod1_attended, mod2_attended], dim=-1)   # [B, L, 2*d]
        gate = self.gate(gate_in).softmax(dim=-1)                    # [B, L, 2]

        mixed = (
            gate[..., 0:1] * mod1_attended +
            gate[..., 1:2] * mod2_attended
        )  # [B, L, d_model]
        
        return self.norm(mixed)

class CrossAttentionTwoStreamFusion(nn.Module):
    """
    Generic cross-attention fusion for two token-aligned streams.
    Inputs can be any pair of modalities (e.g., ESM↔Ankh, ProtT5↔PGLM).
    """
    
    def __init__(self,
                 d_in1: int = 1152,
                 d_in2: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        self.d_out = d_out
        
        # Normalization layers
        self.ln_in1 = nn.LayerNorm(d_in1)
        self.ln_in2 = nn.LayerNorm(d_in2)
        
        # Projection layers
        self.proj_in1 = nn.Linear(d_in1, d_out)
        self.proj_in2 = nn.Linear(d_in2, d_out)
        
        # Cross-attention fusion (two-way with gating)
        self.fusion = CrossAttentionFusionTwoModalities(d_out, dropout, n_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, emb1, emb2, pad_mask=None, lengths=None):
        # Normalize and project
        emb1_proj = self.proj_in1(self.ln_in1(emb1))  # [B, L, d_out]
        emb2_proj = self.proj_in2(self.ln_in2(emb2))  # [B, L, d_out]
        
        # Apply cross-attention fusion
        fused = self.fusion(emb1_proj, emb2_proj, pad_mask)
        
        return self.dropout(fused).masked_fill(pad_mask.unsqueeze(-1), 0.0)

# Drop-in replacement for your current SequenceEncoder
class CrossAttentionSequenceEncoder(nn.Module):
    def __init__(self,
                 d_in1: int = 1152,
                 d_in2: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        
        self.fusion = CrossAttentionTwoStreamFusion(
            d_in1=d_in1,
            d_in2=d_in2,
            d_out=d_out,
            dropout=dropout,
            n_heads=n_heads
        )
    
    def forward(self, emb1, emb2, pad_mask=None, lengths=None):
        return self.fusion(emb1, emb2, pad_mask)


# NOTE: The specialized Prot/PGLM fusion has been unified into the generic
# CrossAttentionTwoStreamFusion via CrossAttentionSequenceEncoder.

############################################################
#  Dual-stream cross-attention fusion
############################################################

class DualStreamCrossAttention(nn.Module):
    def __init__(self, d_model: int = 1152, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "left_norm": nn.LayerNorm(d_model),
                "right_norm": nn.LayerNorm(d_model),
                "left_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "right_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "left_ffn": ResidualFeedForward(d_model, dropout=dropout),
                "right_ffn": ResidualFeedForward(d_model, dropout=dropout),
            })
            self.layers.append(layer)

    def forward(
        self, 
        left: torch.Tensor, 
        right: torch.Tensor,
        left_pad: Optional[torch.Tensor] = None,  # [B, L] True where padded
        right_pad: Optional[torch.Tensor] = None   # [B, L] True where padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            # left attends to right
            left2, _ = layer["left_attn"](
                layer["left_norm"](left),
                layer["right_norm"](right),
                layer["right_norm"](right),
                key_padding_mask=right_pad,
            )
            # right attends to left
            right2, _ = layer["right_attn"](
                layer["right_norm"](right),
                layer["left_norm"](left),
                layer["left_norm"](left),
                key_padding_mask=left_pad,
            )
            left, right = left + left2, right + right2
            left = layer["left_ffn"](left)
            right = layer["right_ffn"](right)
        return left, right



############################################################
#  LightningModule - Optimized Implementation
############################################################

class ProteinLitModule(LightningModule):
    def __init__(
        self,
        task_type: str,                       # Task type: "mf", "bp", or "cc" - required
        d_model: int = 1152,                  # Base model dimension
        d_prot: int = 1024,                   # ProtT5 hidden size (XL UniRef50 = 1024)
        d_ankh: int = 1024,                   # Ankh3-Large embedding dimension
        d_pglm: int = 1536,                    # PGLM embedding dimension
        n_cross_layers: int = 2,              # Cross-attention layers
        n_heads: int = 8,                     # Attention heads
        dropout: float = 0.1,                 # Dropout rate
        optimizer: Optional[Any] = None,      # Hydra optimizer config
        scheduler: Optional[Any] = None,      # Hydra scheduler config  
        warmup_ratio: float = 0.05,           # Warmup ratio for cosine schedule (5% of total training steps)
        lr_main: float = 1e-4,                # LR for fusion/head params (LoRA pairs well with higher LR here)
        lr_plm: float = 1e-4,                 # LR for LoRA params inside PLMs
        # --- LoRA knobs (best-practice defaults) ---
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_include_out: bool = True,        # adapt attention out-proj too
        lora_include_ffn: bool = True,        # adapt FFN up/down projections
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for easy access
        self.task_type = task_type
        
        # Initialize embedding models
        # Replace ESM-C helpers with ESM++ (HF). Hidden size = 1152 (large).
        self.esmpp_model = AutoModelForMaskedLM.from_pretrained(
            "Synthyra/ESMplusplus_large", trust_remote_code=True
        )
        # --- Attach LoRA adapters to *all* encoder layers (validated names) ---
        self.esmpp_model = self._attach_lora_exact(
            self.esmpp_model,
            targets=(["attn.layernorm_qkv.1"] +
                     (["attn.out_proj"] if self.hparams.lora_include_out else []) +
                     (["ffn.1", "ffn.3"] if self.hparams.lora_include_ffn else [])),
            r=self.hparams.lora_r,
            alpha=self.hparams.lora_alpha,
            dropout=self.hparams.lora_dropout,
        )

        # Verify LoRA attachments across all targeted layers
        try:
            # Only verify ESM++ layers since ProtT5 is now pre-computed
            qkv = count_hits(self.esmpp_model, "attn.layernorm_qkv.1.lora_A")
            print("ESM++ QKV LoRA blocks:", qkv, f"(expected 36)")
            if self.hparams.lora_include_out:
                outp = count_hits(self.esmpp_model, "attn.out_proj.lora_A")
                print("ESM++ out_proj LoRA blocks:", outp, f"(expected 36)")
            if self.hparams.lora_include_ffn:
                up = count_hits(self.esmpp_model, "ffn.1.lora_A")
                dn = count_hits(self.esmpp_model, "ffn.3.lora_A")
                print("ESM++ FFN up LoRA blocks:", up, f"(expected 36)")
                print("ESM++ FFN dn LoRA blocks:", dn, f"(expected 36)")
        except Exception as e:
            print(f"LoRA verification failed with error: {e}")
 
        
        # NOTE: MSA model removed; MSA tokens may still be passed but are unused
        
        # Individual learnable BOS/EOS tokens for each modality
        # ESM-C tokens (d_model dimension)
        self.esm_bos_token = nn.Parameter(torch.zeros(1, d_model))
        self.esm_eos_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.esm_bos_token, std=0.02)
        nn.init.normal_(self.esm_eos_token, std=0.02)
        
        # ProtT5 tokens (d_prot dimension) - for pre-computed embeddings
        self.prot_bos_token = nn.Parameter(torch.zeros(1, d_prot))  # ProtT5 dim
        self.prot_eos_token = nn.Parameter(torch.zeros(1, d_prot))  # ProtT5 dim
        nn.init.normal_(self.prot_bos_token, std=0.02)
        nn.init.normal_(self.prot_eos_token, std=0.02)
        
        # Ankh tokens (d_ankh dimension)
        self.ankh_bos_token = nn.Parameter(torch.zeros(1, d_ankh))
        self.ankh_eos_token = nn.Parameter(torch.zeros(1, d_ankh))
        nn.init.normal_(self.ankh_bos_token, std=0.02)
        nn.init.normal_(self.ankh_eos_token, std=0.02)
        
        # PGLM tokens (use d_msa argument to represent pglm embedding dim)
        self.pglm_bos_token = nn.Parameter(torch.zeros(1, d_pglm))
        self.pglm_eos_token = nn.Parameter(torch.zeros(1, d_pglm))
        nn.init.normal_(self.pglm_bos_token, std=0.02)
        nn.init.normal_(self.pglm_eos_token, std=0.02)
        
        # Encoders
        # Sequence stream: fuse ESM-C with Ankh
        self.seq_encoder = CrossAttentionSequenceEncoder(
            d_in1=d_model,
            d_in2=d_ankh,
            d_out=768,
            dropout=dropout,
            n_heads=n_heads
        )
        # Second stream: ProtT5 + PGLM fusion (generic two-stream)
        self.prot_pglm_encoder = CrossAttentionSequenceEncoder(
            d_in1=1024,
            d_in2=d_pglm,
            d_out=768,
            dropout=dropout,
            n_heads=n_heads
        )

        # Fusion / Cross-attention
        self.cross_attn = DualStreamCrossAttention(
            d_model=768,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout
        )

        # Use masked mean pooling (function called directly in forward)
        
        # Replace fusion_proj with AttentionFusion
        self.fusion = AttentionFusion(d_model=768)
        
        # Create classifier head
        self.head = MLPHead(768, get_num_classes_for_task(task_type), dropout)

        # For multi-label protein prediction, we use standard BCE loss
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # Storage for epoch-wise sweep for CAFA metrics
        self._val_logits = []
        self._val_labels = []
        self._test_logits = []
        self._test_labels = []

    def setup(self, stage: str) -> None:
        """No-op setup (MSA has been fully removed)."""
        return

    def compute_esmc_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """Compute ESM++ embeddings for a batch (grad-connected).
        Shape: [B, L_max, 1152]; we build a [CLS, x1..xL, EOS] frame and pad."""
        max_len = max(len(s) for s in sequences) + 2  # +2 for CLS/EOS
        tok = self.esmpp_model.tokenizer(
            sequences, padding=True, return_tensors="pt"
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.esmpp_model(**tok)
        hs = out.last_hidden_state  # [B, L_tok, 1152], attached to backbone
        batch_embeddings: List[torch.Tensor] = []
        for i, seq in enumerate(sequences):
            L = len(seq)
            emb = torch.zeros(L + 2, 1152, dtype=hs.dtype, device=hs.device)
            emb[1:L+1] = hs[i, 1:L+1]                    # place only residues
            # EOS slot (L+1) left zeros; learnable EOS gets inserted later
            if emb.size(0) < max_len:
                emb = F.pad(emb, (0, 0, 0, max_len - emb.size(0)), value=0.0)
            batch_embeddings.append(emb)
        return torch.stack(batch_embeddings)           # [B, L_max, 1152]
   


    # ---------------------------------------------------------------------
    #  Forward pass
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with individual learnable BOS/EOS tokens for each modality.
        
        Key changes:
        - Each modality gets its own learnable BOS/EOS tokens
        - BOS tokens are inserted at position 0, EOS tokens at position L+1
        - This allows each modality to learn optimal start/end representations
        """
        # Extract inputs - most embeddings are pre-computed
        sequences = batch["sequence"]    # List of sequences
        ankh_emb = batch["ankh_emb"]     # Pre-computed Ankh3-Large: [B, L, d_ankh]
        prot_emb = batch["prot_emb"]     # Pre-computed ProtT5: [B, L, d_prot]
        pglm_emb = batch["pglm_emb"]     # Pre-computed PGLM: [B, L, d_pglm]
        
        # Compute ESM++ embeddings on-the-fly
        seq_emb = self.compute_esmc_embeddings(sequences)  # [B, L, 1152]

        pad_mask = batch["pad_mask"]     # [B, L] - True where padded
        # no sequence-of-sequences mask needed (no MSA)

        # Skip MSA embedding computation; tokens remain unused but are kept in the batch

        # Insert individual learnable BOS/EOS tokens for each modality
        B = seq_emb.size(0)
        lengths = batch["lengths"]
        
        # Insert BOS tokens at position 0
        seq_emb[:, 0] = self.esm_bos_token.to(seq_emb.dtype).expand(B, -1)      # [B, d_model]
        prot_emb[:, 0] = self.prot_bos_token.to(prot_emb.dtype).expand(B, -1)    # [B, d_prot]
        ankh_emb[:, 0] = self.ankh_bos_token.to(ankh_emb.dtype).expand(B, -1)    # [B, d_ankh]
        pglm_emb[:, 0] = self.pglm_bos_token.to(pglm_emb.dtype).expand(B, -1)    # [B, d_pglm]
        

        
        # Insert EOS tokens at position L+1
        for i, length in enumerate(lengths):
            eos_pos = length + 1
            if eos_pos < seq_emb.size(1):  # Safety check
                seq_emb[i, eos_pos] = self.esm_eos_token.to(seq_emb.dtype).squeeze()      # [d_model]
                prot_emb[i, eos_pos] = self.prot_eos_token.to(prot_emb.dtype).squeeze()    # [d_prot]
                ankh_emb[i, eos_pos] = self.ankh_eos_token.to(ankh_emb.dtype).squeeze()    # [d_ankh]
                pglm_emb[i, eos_pos] = self.pglm_eos_token.to(pglm_emb.dtype).squeeze()    # [d_pglm]
        
        # Create mask for pooling that masks BOS/EOS positions
        pool_mask = pad_mask.clone()
        pool_mask[:, 0] = True  # Mask out BOS position
        
        # Also ensure EOS positions (L+1) are masked as True
        for i, length in enumerate(lengths):
            eos_pos = length + 1
            if eos_pos < pool_mask.size(1):  # Safety check
                pool_mask[i, eos_pos] = True  # Mask out EOS position
        
        # Encode each modality with appropriate masks
        # Sequence stream: ESM++ + Ankh
        seq_z = self.seq_encoder(seq_emb, ankh_emb, pad_mask=pad_mask)  # [B, L, 768]

        # Apply ProtT5 + PGLM cross-modal fusion
        prot_pglm_z = self.prot_pglm_encoder(prot_emb, pglm_emb, pad_mask=pad_mask)  # [B, L, 768]

        # Cross-modal attention with padding masks between the two streams
        # seq_z contains ESM++ + Ankh fusion, prot_pglm_z contains ProtT5 + PGLM fusion
        seq_z, prot_pglm_z = self.cross_attn(seq_z, prot_pglm_z, pad_mask, pad_mask)  # each [B, L, d]

        # Use masked mean pooling (exclude BOS/EOS tokens)
        seq_pool = masked_mean_pooling(seq_z, pool_mask)  # [B, d] - ESM++ + Ankh stream
        aux_pool = masked_mean_pooling(prot_pglm_z, pool_mask)  # [B, d] - ProtT5 + PGLM stream

        # Use AttentionFusion between the two streams
        fused = self.fusion(seq_pool, aux_pool)    # [B, d]
        
        logits = self.head(fused)                  # [B, C]
        
        return logits, batch["labels"]

    # All MSA-related utilities removed



    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update running mean metric
        self.train_loss(loss)

        # Log the raw loss for the current step (avoid computing MeanMetric prematurely)
        self.log("train/loss_step", loss, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log learning rates for PLM and non-PLM parameter groups every step.

        Group 0: PLM params (ESM++ only)
        Group 1: Main params (fusion layers, heads, etc.)
        """
        if not self.trainer.optimizers:
            return
        optimizer = self.trainer.optimizers[0]
        param_groups = getattr(optimizer, "param_groups", [])
        if len(param_groups) >= 2:
            plm_lr = float(param_groups[0]["lr"])
            main_lr = float(param_groups[1]["lr"])
            self.log("lr/plm", plm_lr, on_step=True, prog_bar=True, sync_dist=True)
            self.log("lr/main", main_lr, on_step=True, prog_bar=True, sync_dist=True)
        elif len(param_groups) == 1:
            # Fallback if only one group exists
            group0_lr = float(param_groups[0]["lr"])
            self.log("lr/group0", group0_lr, on_step=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.val_loss(loss)

        # Store for epoch-wise CAFA metrics computation
        # Convert to fp32 before CPU transfer to avoid bf16 → numpy issues
        self._val_logits.append(logits.detach().float().cpu())   # keep raw, no sigmoid
        self._val_labels.append(labels.detach().float().cpu())
        
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.test_loss(loss)
 
        # Store for epoch-wise CAFA metrics computation
        # Convert to fp32 before CPU transfer to avoid bf16 → numpy issues
        self._test_logits.append(logits.detach().float().cpu())   # keep raw, no sigmoid
        self._test_labels.append(labels.detach().float().cpu())

    def _heal_metrics(self, logits, labels): 
        probs = torch.sigmoid(logits).numpy()
        labels = labels.numpy().astype(int)
        goterms = list(self.trainer.datamodule._go_dicts[self.task_type].keys())

        # Parent-child propagation
        if self.task_type == "ec": #NOTE: If ec is enabled update this
            print("EC task type detected; ENSURE UPDATE")
            probs = propagate_ec_preds(probs, goterms)
        else:
            probs = propagate_go_preds(probs, goterms)

        macro, micro = function_centric_aupr(labels, probs)
        fmax, _      = cafa_fmax(labels, probs, goterms, self.task_type)
        s_min        = smin(labels, probs, self.trainer.datamodule.ic_vector.numpy())

        return macro, micro, fmax, s_min

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits)
        labels = torch.cat(self._val_labels)
        macro, micro, fmax, s_min = self._heal_metrics(logits, labels)

        # Convert numpy scalars to Python floats for logging
        self.log_dict({
            "val/loss": float(self.val_loss.compute()),
            "val/AUPR_macro": float(macro),
            "val/AUPR_micro": float(micro), 
            "val/Fmax": float(fmax),
            "val/Smin": float(s_min),
        }, prog_bar=True, sync_dist=True)

        self.val_loss.reset(); self._val_logits.clear(); self._val_labels.clear()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset validation metrics to avoid storing results from sanity checks
        self.val_loss.reset()
        self._val_logits.clear()
        self._val_labels.clear()
        # Log parameter group sizes and initial learning rates for verification
        plm_params, main_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("esmpp_model"):
                plm_params.append(p)
            else:
                main_params.append(p)
        num_plm = sum(p.numel() for p in plm_params)
        num_main = sum(p.numel() for p in main_params)
        
        # Print parameter counts to console
        print(f"Number of PLM parameters: {num_plm:,}")
        print(f"Number of main parameters: {num_main:,}")
        print(f"Total trainable parameters: {num_plm + num_main:,}")
        
        self.log("params/num_plm_params", float(num_plm), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("params/num_main_params", float(num_main), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # If optimizer is initialized, log initial LRs
        if self.trainer and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            param_groups = getattr(optimizer, "param_groups", [])
            if len(param_groups) >= 2:
                self.log("lr_initial/plm", float(param_groups[0]["lr"]), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
                self.log("lr_initial/main", float(param_groups[1]["lr"]), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
    
    # --- NEW: LoRA attachment helper with exact target substrings ---
    def _attach_lora_exact(self, model: nn.Module, targets: List[str], r: int, alpha: int, dropout: float) -> nn.Module:
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        wrapped = get_peft_model(model, cfg)
        wrapped.print_trainable_parameters()
        return wrapped
       
    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            raise ValueError("Optimizer must be provided in hparams")
        # Two groups: (0) LoRA params inside ESM++, (1) fusion/head/etc.
        plm_params, main_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("esmpp_model"):
                plm_params.append(p)
            else:
                main_params.append(p)
        param_groups = [
            {"params": plm_params, "lr": float(self.hparams.lr_plm),  "weight_decay": 0.0},
            {"params": main_params, "lr": float(self.hparams.lr_main), "weight_decay": 0.01},
        ]
        optimizer = self.hparams.optimizer(params=param_groups)

        if self.hparams.scheduler is not None:
            # Get total steps
            total_steps = self.trainer.estimated_stepping_batches
            print(f"Total steps: {total_steps}")
            # Warmup steps
            warmup_steps = int(self.hparams.warmup_ratio * total_steps)
            scheduler_fn = self.hparams.scheduler 
            scheduler = scheduler_fn(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # Log aggregated training loss _after_ all updates have occurred this epoch
        self.log("train/loss_epoch", self.train_loss.compute(), prog_bar=True, sync_dist=True)
        # Reset for next epoch
        self.train_loss.reset()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        logits = torch.cat(self._test_logits)
        labels = torch.cat(self._test_labels)
        macro, micro, fmax, s_min = self._heal_metrics(logits, labels)

        # Convert numpy scalars to Python floats for logging
        self.log_dict({
            "test/loss": float(self.test_loss.compute()),
            "test/AUPR_macro": float(macro),
            "test/AUPR_micro": float(micro),
            "test/Fmax": float(fmax),
            "test/Smin": float(s_min),
        }, prog_bar=True, sync_dist=True)

        self.test_loss.reset(); self._test_logits.clear(); self._test_labels.clear()