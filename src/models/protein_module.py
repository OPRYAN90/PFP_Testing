from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_out)
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
    
    def __init__(self, d_model: int, attention_dropout: float = 0.1, n_heads: int = 8, fusion_mlp_dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention for each modality pair
        self.mod1_to_mod2 = nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
        self.mod2_to_mod1 = nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
        
        # Per-token gating over {mod1, mod2}
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(fusion_mlp_dropout),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(fusion_mlp_dropout),
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
                 attention_dropout: float = 0.1,
                 fusion_mlp_dropout: float = 0.1,
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
        self.fusion = CrossAttentionFusionTwoModalities(d_out, attention_dropout, n_heads, fusion_mlp_dropout)
        
        self.dropout = nn.Dropout(fusion_mlp_dropout)
    
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
                 attention_dropout: float = 0.1,
                 fusion_mlp_dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        
        self.fusion = CrossAttentionTwoStreamFusion(
            d_in1=d_in1,
            d_in2=d_in2,
            d_out=d_out,
            attention_dropout=attention_dropout,
            fusion_mlp_dropout=fusion_mlp_dropout,
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
    def __init__(self, d_model: int = 1152, n_heads: int = 8, n_layers: int = 2, attention_dropout: float = 0.1, ffn_dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "left_norm": nn.LayerNorm(d_model),
                "right_norm": nn.LayerNorm(d_model),
                "left_attn": nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True),
                "right_attn": nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True),
                "left_ffn": ResidualFeedForward(d_model, dropout=ffn_dropout),
                "right_ffn": ResidualFeedForward(d_model, dropout=ffn_dropout),
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
        d_latent: int = 768,                  # Latent dimension for cross-modal fusion and final representation
        d_prot: int = 1024,                   # ProtT5 hidden size (XL UniRef50 = 1024)
        d_ankh: int = 1024,                   # Ankh3-Large embedding dimension
        d_pglm: int = 1536,                    # PGLM embedding dimension
        n_cross_layers: int = 2,              # Cross-attention layers
        n_heads: int = 8,                     # Attention heads
        attention_dropout: float = 0.14,      # For all attention modules
        ffn_dropout: float = 0.20,            # For FFNs (ResidualFeedForward)
        fusion_mlp_dropout: float = 0.05,     # For fusion/gating MLPs
        head_dropout: float = 0.30,           # For final head MLP
        optimizer: Optional[Any] = None,      # Hydra optimizer config
        scheduler: Optional[Any] = None,      # Hydra scheduler config  
        warmup_ratio: float = 0.05,           # Warmup ratio for cosine schedule (5% of total training steps)
        learning_rate: float = 1e-4,          # Learning rate for all parameters
        # --- SupCon knobs (single-GPU friendly) ---
        supcon_on: bool = True,
        supcon_lambda: float = 0.2,      # strength of SupCon vs BCE (tune 0.1–0.4)
        supcon_tau: float = 0.12,        # temperature (try 0.05–0.12)
        supcon_proj_dim: int = 256,      # contrastive embedding dim
        supcon_proj_dropout: float = 0.10,
        supcon_queue_size: int = 4096,   # 0 disables memory queue
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for easy access
        self.task_type = task_type
 
        
        # NOTE: MSA model removed; MSA tokens may still be passed but are unused
        
        # Individual learnable BOS/EOS tokens for each modality
        # ESM-C tokens (d_model dimension) - for pre-computed embeddings
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
            d_out=d_latent,
            attention_dropout=attention_dropout,
            fusion_mlp_dropout=fusion_mlp_dropout,
            n_heads=n_heads
        )
        # Second stream: ProtT5 + PGLM fusion (generic two-stream)
        self.prot_pglm_encoder = CrossAttentionSequenceEncoder(
            d_in1=1024,
            d_in2=d_pglm,
            d_out=d_latent,
            attention_dropout=attention_dropout,
            fusion_mlp_dropout=fusion_mlp_dropout,
            n_heads=n_heads
        )

        # Fusion / Cross-attention
        self.cross_attn = DualStreamCrossAttention(
            d_model=d_latent,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout
        )

        # Use masked mean pooling (function called directly in forward)
        
        # Replace fusion_proj with AttentionFusion
        self.fusion = AttentionFusion(d_model=d_latent, p_drop=fusion_mlp_dropout)
        
        # --- NEW: SupCon projection head (single pass) ---
        self.supcon_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent*2),
            nn.GELU(),
            nn.Dropout(self.hparams.supcon_proj_dropout),
            nn.Linear(d_latent*2, self.hparams.supcon_proj_dim),
        )

        # --- NEW: XBM-style memory queue (single-GPU) ---
        Q = int(self.hparams.supcon_queue_size)
        D = int(self.hparams.supcon_proj_dim)
        C = get_num_classes_for_task(task_type)
        if Q > 0:
            # Store bank in float32 for higher-precision similarity math
            self.register_buffer("queue_z", torch.zeros(Q, D, dtype=torch.float32))
            self.register_buffer("queue_y", torch.zeros(Q, C, dtype=torch.uint8))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_count", torch.zeros(1, dtype=torch.long))

        # Create classifier head
        self.head = MLPHead(d_latent, get_num_classes_for_task(task_type), head_dropout)

        # For multi-label protein prediction, we use standard BCE loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        # TODO: For long-tail multi-label, try ASL or logit-adjusted BCE as a drop-in.

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

    # ---- NEW: helper to select the "shared trunk" params (exclude heads) ----
    def _shared_params(self):
        # Keeps it cheap: skip classifier head and SupCon projection head.
        return tuple(
            p for n, p in self.named_parameters()
            if p.requires_grad and not (n.startswith("head.") or n.startswith("supcon_head."))
        )


    # ---------------------------------------------------------------------
    #  Forward pass
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any], want_fused: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with individual learnable BOS/EOS tokens for each modality.
        
        Key changes:
        - Each modality gets its own learnable BOS/EOS tokens
        - BOS tokens are inserted at position 0, EOS tokens at position L+1
        - This allows each modality to learn optimal start/end representations
        
        Args:
            batch: Input batch dictionary
            want_fused: If True, also return the fused representation for gradient probing
        """
        # Extract inputs - all embeddings are pre-computed
        seq_emb = batch["esmc_emb"]      # Pre-computed ESM-C: [B, L, d_model]
        ankh_emb = batch["ankh_emb"]     # Pre-computed Ankh3-Large: [B, L, d_ankh]
        prot_emb = batch["prot_emb"]     # Pre-computed ProtT5: [B, L, d_prot]
        pglm_emb = batch["pglm_emb"]     # Pre-computed PGLM: [B, L, d_pglm]

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

        # --- NEW: contrastive embedding (single pass) ---
        z = F.normalize(self.supcon_head(fused), dim=-1)  # [B, supcon_proj_dim]

        logits = self.head(fused)                  # [B, C]
        if want_fused:
            return logits, batch["labels"], z, fused
        return logits, batch["labels"], z

    # All MSA-related utilities removed



    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # keep want_fused=True to preserve interface; fused no longer used for grad probe
        logits, labels, z, fused = self.forward(batch, want_fused=True)

        loss_bce = self.loss_fn(logits, labels)
        log_dict = {"train/bce_loss": loss_bce}

        if self.hparams.supcon_on:
            loss_sup = self._supcon_loss(z, labels)
            log_dict.update({"train/supcon": loss_sup})

            # ---- PARAMETER-SPACE grad probe on last shared params ----
            shared = self._shared_params()

            # grads wrt shared params (lightweight; no accumulation to .grad)
            g_bce_params = torch.autograd.grad(
                loss_bce, shared, retain_graph=True, allow_unused=True
            )
            g_sup_params = torch.autograd.grad(
                loss_sup, shared, retain_graph=True, allow_unused=True
            )

            # global L2 norms in fp32
            def _l2(gs):
                acc = None
                for g in gs:
                    if g is None:
                        continue
                    v = g.detach().float().reshape(-1)
                    acc = v if acc is None else torch.cat([acc, v], dim=0)
                return (acc.pow(2).sum() + 1e-20).sqrt() if acc is not None else torch.tensor(0.0, device=loss_bce.device)

            gn_bce = _l2(g_bce_params)
            gn_sup = _l2(g_sup_params)

            # cosine between tasks in PARAM space
            dot = torch.tensor(0.0, device=loss_bce.device, dtype=torch.float32)
            for a, b in zip(g_bce_params, g_sup_params):
                if a is None or b is None:
                    continue
                dot = dot + (a.detach().float() * b.detach().float()).sum()
            cos = dot / (gn_bce * gn_sup + 1e-20)

            # target gradient share r_tgt := supcon_lambda (e.g., 0.20)
            r_tgt = float(self.hparams.supcon_lambda)
            scale = (r_tgt * gn_bce / (gn_sup + 1e-12)).clamp(0.0, 5.0).detach()

            # soft conflict gate: suppress when anti-aligned
            gate = (cos >= 0).to(cos.dtype).detach()  # binary gate: 1 if cos≥0 else 0

            # combined loss with adaptive scaling + gating
            loss = loss_bce + gate * scale * loss_sup

            # logs (keep old keys stable + add a few new)
            log_dict.update({
                "train/gn_bce": gn_bce,
                "train/gn_sup": gn_sup,
                "train/grad_ratio_sup_over_bce": (gn_sup / (gn_bce + 1e-12)).detach(),
                "train/grad_cos_param": cos,
                "train/scale": scale,
                "train/gate": gate,
                "train/grad_ratio_eff": (scale * (gn_sup / (gn_bce + 1e-12))).detach(),
            })
        else:
            loss = loss_bce

        if self.hparams.supcon_on:
            with torch.no_grad():
                self._enqueue(z, labels)

        self.train_loss(loss)
        self.log_dict(log_dict, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_step", loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log learning rate every step."""
        if not self.trainer.optimizers:
            return
        optimizer = self.trainer.optimizers[0]
        param_groups = getattr(optimizer, "param_groups", [])
        if len(param_groups) >= 1:
            lr = float(param_groups[0]["lr"])
            self.log("lr", lr, on_step=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels, _ = self.forward(batch, want_fused=False)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.val_loss(loss)

        # Store for epoch-wise CAFA metrics computation
        # Convert to fp32 before CPU transfer to avoid bf16 → numpy issues
        self._val_logits.append(logits.detach().float().cpu())   # keep raw, no sigmoid
        self._val_labels.append(labels.detach().float().cpu())
        
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels, _ = self.forward(batch, want_fused=False)
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
        # Log parameter counts for verification
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Print parameter counts to console
        print(f"Total trainable parameters: {total_params:,}")
        
        self.log("params/total_params", float(total_params), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # If optimizer is initialized, log initial LR
        if self.trainer and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            param_groups = getattr(optimizer, "param_groups", [])
            if len(param_groups) >= 1:
                self.log("lr_initial", float(param_groups[0]["lr"]), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
    

       
    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            raise ValueError("Optimizer must be provided in hparams")
        # Single parameter group with unified learning rate
        optimizer = self.hparams.optimizer(params=self.parameters())

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

    # ------------------------------------------------------------------
    #  NEW: SupCon helpers (queue + multi-label weighted SupCon)
    # ------------------------------------------------------------------
    def _enqueue(self, z: torch.Tensor, y: torch.Tensor):
        """Push current mini-batch (detached) into the memory queue."""
        Q = int(self.hparams.supcon_queue_size)
        if Q <= 0:
            return
        # Keep queue precision as float32
        z = z.detach().to(self.queue_z.dtype)
        y = (y.detach() > 0).to(torch.uint8)

        B = z.size(0)
        ptr = int(self.queue_ptr.item())
        end = ptr + B

        if end <= Q:
            self.queue_z[ptr:end] = z
            self.queue_y[ptr:end] = y
        else:
            first = Q - ptr
            self.queue_z[ptr:] = z[:first]
            self.queue_y[ptr:] = y[:first]
            remain = B - first
            if remain > 0:
                self.queue_z[:remain] = z[first:first+remain]
                self.queue_y[:remain] = y[first:first+remain]

        self.queue_ptr[0] = (ptr + B) % Q
        # TODO: assign with Python int for absolute safety (minor): self.queue_count[0] = min(Q, int(self.queue_count.item()) + B)
        self.queue_count[0] = torch.clamp(self.queue_count + B, max=Q)

    def _supcon_loss(self, z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Multi-label SupCon with Jaccard positive weights.
        z: [B,D] (L2-normalized); Y: [B,C] multi-hot.
        Uses candidate set = [queue (if any) ; current batch].
        """
        tau = float(self.hparams.supcon_tau)

        # --- Force high-precision compute in this block ---
        ac_dev = "cuda" if z.is_cuda else "cpu"
        # Disable autocast so matmuls/log-sum-exp run in fp32 reliably.
        with torch.autocast(device_type=ac_dev, enabled=False):
            z = z.float()
            Y = Y.float()

            # Build candidate bank
            Q = int(self.hparams.supcon_queue_size)
            if Q > 0 and int(self.queue_count.item()) > 0:
                k = int(self.queue_count.item())
                Z_all = torch.cat([self.queue_z[:k].float(), z], dim=0)                 # [N,D] fp32
                Y_all = torch.cat([self.queue_y[:k].to(torch.uint8), (Y > 0).to(torch.uint8)], dim=0)  # [N,C]
                offset = Z_all.size(0) - z.size(0)  # start of in-batch region
            else:
                Z_all = z
                Y_all = (Y > 0).to(torch.uint8)
                offset = 0

            # Cosine similarities / logits (fp32).
            logits = (z @ Z_all.T) / tau                                          # [B,N]
            logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
            # REQUIRED: mask self FIRST (works with or without queue)
            B = z.size(0)
            idx = torch.arange(B, device=z.device)
            self_idx = (offset + idx) if offset > 0 else idx
            logits[idx, self_idx] = float("-inf")
            # Numerically safe normalization: guard rows that are all -inf
            den = torch.logsumexp(logits, dim=1, keepdim=True)                    # [B,1]
            log_prob = logits - den                                               # [B,N]

            # ---- Positive weights: Jaccard ----
            Yb = (Y > 0).to(torch.float32)                        # [B,C]
            Yall = Y_all.to(torch.float32)                        # [N,C]
            inter = Yb @ Yall.T                                   # [B,N]
            pos_mask = inter > 0
            # REQUIRED: exclude the anchor itself from the positive set
            if Z_all.size(0) >= (offset + B):
                pos_mask[idx, self_idx] = False

            # Jaccard similarity weights
            sum_i = Yb.sum(dim=1, keepdim=True)                           # [B,1]
            sum_j = Yall.sum(dim=1, keepdim=True).T                       # [1,N]
            union = (sum_i + sum_j - inter).clamp_min(1)
            W = (inter / union) * pos_mask.float()

            # after W is built and pos_mask excludes self
            safe_log_prob = log_prob.masked_fill(~pos_mask, 0.0)   # restrict to P(i)

            # Weighted average over positives per anchor
            Wsum = W.sum(dim=1)                                              # [B]
            valid = Wsum > 0
            if valid.any():
                pos_log_prob = (W[valid] * safe_log_prob[valid]).sum(dim=1) / Wsum[valid].clamp_min(1e-6)
                return -pos_log_prob.mean()
            else:
                return logits.new_zeros(())