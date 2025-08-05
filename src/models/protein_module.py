from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# Optional FlashAttention toggle – does nothing on unsupported systems
# -----------------------------------------------------------------------------

# from utils.flash_control import maybe_enable_flash_attention  
# maybe_enable_flash_attention(True)


from lightning import LightningModule
from torchmetrics import MeanMetric
import numpy as np
from data.go_utils import (
    propagate_go_preds, propagate_ec_preds,
    function_centric_aupr, cafa_fmax, smin
)

# Simple helper to get number of classes without datamodule dependency
def get_num_classes_for_task(task_type: str) -> int:
    """Get number of classes for a task type."""
    class_counts = {"mf": 489, "bp": 1943, "cc": 320}
    return class_counts[task_type]

class ResidualFeedForward(nn.Module):
    """Transformer-style position-wise FFN with residual + layer norm."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),  # Standard: dropout after activation
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),  # Standard: dropout before residual
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            nn.Linear(602, d_out), #EXTREME WARNING NOTE: DIMS DEPEDNING ON TASK AND D_MSA
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
    """
    Lightweight fusion: a **scalar gate** decides how much to trust each
    modality; a single linear layer adds extra capacity.
    """
    def __init__(self, d_model: int = 768):
        super().__init__()

        # Scalar gate in [0,1]
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 1, bias=True),
            nn.Sigmoid()
        )


    def forward(
        self,
        seq_feat: torch.Tensor,          # [B, D]
        msa_feat: torch.Tensor,          # [B, D]
    ) -> torch.Tensor:

        concat = torch.cat([seq_feat, msa_feat], dim=-1)             # [B, 2D]

        g = self.gate(concat)                                       # [B,1]

        fused = g * seq_feat + (1.0 - g) * msa_feat                  # gated sum
        return fused


############################################################
#  Encoders for the three modalities
############################################################

class SeparateMHACrossAttentionFusion(nn.Module):
    """Cross-attention with separate MHA for each modality pair + softmax gating"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # (unchanged) 6 MHAs
        self.esm_to_prot = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.esm_to_ankh = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prot_to_esm = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prot_to_ankh = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ankh_to_esm = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ankh_to_prot = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # NEW: per-token gating over {esm, prot, ankh}
        self.gate = nn.Sequential(
        nn.LayerNorm(d_model * 3),
        nn.Dropout(0.1),                # (optional) on inputs to the gate
        nn.Linear(d_model * 3, d_model // 2),
        nn.GELU(),
        nn.Dropout(0.1),                # on hidden activations
        nn.Linear(d_model // 2, 3)      # logits -> softmax outside
        )

        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, esm, prot, ankh, pad_mask=None):
        # (unchanged) optional pre-mask of queries
        if pad_mask is not None:
            esm  = esm.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            prot = prot.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            ankh = ankh.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        kpm = pad_mask  # MultiheadAttention expects True==pad

        # (unchanged) 6 cross attentions
        esm_to_prot_att, _  = self.esm_to_prot(esm,  prot, prot, key_padding_mask=kpm)
        esm_to_ankh_att, _  = self.esm_to_ankh(esm,  ankh, ankh, key_padding_mask=kpm)
        prot_to_esm_att, _  = self.prot_to_esm(prot, esm,  esm,  key_padding_mask=kpm)
        prot_to_ankh_att, _ = self.prot_to_ankh(prot, ankh, ankh, key_padding_mask=kpm)
        ankh_to_esm_att, _  = self.ankh_to_esm(ankh, esm,  esm,  key_padding_mask=kpm)
        ankh_to_prot_att, _ = self.ankh_to_prot(ankh, prot, prot, key_padding_mask=kpm)

        # (unchanged) residual aggregation per stream
        esm_att  = esm  + esm_to_prot_att  + esm_to_ankh_att
        prot_att = prot + prot_to_esm_att  + prot_to_ankh_att
        ankh_att = ankh + ankh_to_esm_att  + ankh_to_prot_att

        # NEW: convex (softmax) gating per token over the three streams
        gate_in = torch.cat([esm_att, prot_att, ankh_att], dim=-1)   # [B, L, 3*d]
        gate = self.gate(gate_in).softmax(dim=-1)                    # [B, L, 3]

        mixed = (
            gate[..., 0:1] * esm_att +
            gate[..., 1:2] * prot_att +
            gate[..., 2:3] * ankh_att
        )  # [B, L, d_model]

        return self.norm(mixed)  

class SeparateMHAMultiModalFusion(nn.Module):
    """
    Cross-attention fusion using separate MHA for each modality pair
    """
    
    def __init__(self,
                 d_esm: int = 1152,
                 d_prot: int = 128,
                 d_ankh: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        self.d_out = d_out
        
        # Normalization layers (same as original)
        self.ln_esm = nn.LayerNorm(d_esm)
        self.ln_prot = nn.LayerNorm(d_prot)
        self.ln_ankh = nn.LayerNorm(d_ankh)
        
        # Projection layers (same as original)
        self.proj_esm = nn.Linear(d_esm, d_out)
        self.proj_prot = nn.Linear(d_prot, d_out)
        self.proj_ankh = nn.Linear(d_ankh, d_out)
        
        # Separate MHA cross-attention fusion
        self.fusion = SeparateMHACrossAttentionFusion(d_out, dropout, n_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, esm_emb, prot_emb, ankh_emb, pad_mask=None, lengths=None):
        # Normalize and project (same as original)
        esm_proj = self.proj_esm(self.ln_esm(esm_emb))      # [B, L, d_out]
        prot_proj = self.proj_prot(self.ln_prot(prot_emb))  # [B, L, d_out]
        ankh_proj = self.proj_ankh(self.ln_ankh(ankh_emb))  # [B, L, d_out]
        
        # Apply separate MHA cross-attention fusion
        fused = self.fusion(esm_proj, prot_proj, ankh_proj, pad_mask)
        
        # Preserve special tokens (same as original)
        if lengths is not None:
            fused[:, 0] = esm_proj[:, 0]  # Keep BOS token
            for i, length in enumerate(lengths):
                eos_pos = length + 1
                if eos_pos < fused.size(1):  # Safety check
                    fused[i, eos_pos] = esm_proj[i, eos_pos]  # Keep EOS token
        
        return self.dropout(fused).masked_fill(pad_mask.unsqueeze(-1), 0.0)

# Drop-in replacement for comparison
class SeparateMHASequenceEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 1152,
                 d_prot: int = 128,
                 d_ankh: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        
        self.fusion = SeparateMHAMultiModalFusion(
            d_esm=d_model,
            d_prot=d_prot, 
            d_ankh=d_ankh,
            d_out=d_out,
            dropout=dropout,
            n_heads=n_heads
        )
    
    def forward(self, x, prot_emb, ankh_emb, pad_mask=None, lengths=None):
        return self.fusion(x, prot_emb, ankh_emb, pad_mask, lengths)





class MSAEncoder(nn.Module):
    r"""
    Multiple‑Sequence‑Alignment encoder with optional attention‑weighted pooling.

    Inputs
    ------
    msa       : Tensor[B, N_seq, L_pad, d_msa]
    pad_mask  : Tensor[B, L_pad]     – True where residue is padding
    seq_pad   : Tensor[B, N_seq]     – True where whole row is padding

    Output
    ------
    x         : Tensor[B, L_pad, d_msa]
    """

    def __init__(
        self,
        d_msa: int = 768,
        p_chan: float = 0.10,
        p_feat: float = 0.10,
    ):
        super().__init__()
        # 1. Row dropout on MSA embeddings
        self.dropout = nn.Dropout(p_chan)

        # 2. Conservation head for attention scores
        self.conservation_head = nn.Linear(d_msa, 1, bias=False)

        # 3. Learned temperature τ (only used when attention is enabled)
        self.log_tau = nn.Parameter(torch.zeros(()))  # τ = e^{log τ}, init‑1.0

        # 4. Post‑pooling refinement
        self.post_ffn = ResidualFeedForward(d_msa, expansion=4, dropout=p_feat)
        self.norm = nn.LayerNorm(d_msa)

    def forward(
        self,
        msa: torch.Tensor,      # [B, N_seq, L_pad, d_msa]
        pad_mask: torch.Tensor, # [B, L_pad]
        seq_pad: torch.Tensor,  # [B, N_seq]
    ) -> torch.Tensor:

        B, N_seq, L_pad, D = msa.shape
        msa = self.dropout(msa)

        # Build boolean mask [B, N_seq, L_pad]
        pad_bool = seq_pad.unsqueeze(-1) | pad_mask.unsqueeze(1)

        # Count valid rows per example   [B, 1]
        valid_seq = (~seq_pad).sum(dim=1, keepdim=True).clamp(min=1).float()

        # --------------------------------------------------------------------
        # POOLING
        # --------------------------------------------------------------------
        # 1) Conservation logits
        scores = self.conservation_head(msa).squeeze(-1)          # [B,N,L]

        # 2) Centre per column
        scores = scores.masked_fill(pad_bool, 0.0)

        # 3) Scale by (√N · τ)
        scale = (valid_seq.sqrt() * self.log_tau.exp()).unsqueeze(-1)
        scores = scores / scale

        # 4) Mask padding & soft‑max over rows
        scores = scores.masked_fill(pad_bool, -1e4)
        weights = torch.softmax(scores, dim=1).masked_fill(pad_bool, 0.0)

        # 5) Weighted sum
        pooled = (msa * weights.unsqueeze(-1)).sum(dim=1)         # [B,L,D]
        # --------------------------------------------------------------------
        # Refinement & output
        # --------------------------------------------------------------------
        x = self.norm(pooled)
        x = self.post_ffn(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


############################################################
#  Cross-modal attention fusion
############################################################

class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int = 1152, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        print(d_model)
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "seq_norm": nn.LayerNorm(d_model),
                "msa_norm": nn.LayerNorm(d_model),
                "seq_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "msa_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "seq_ffn": ResidualFeedForward(d_model, dropout=dropout),
                "msa_ffn": ResidualFeedForward(d_model, dropout=dropout),
            })
            self.layers.append(layer)

    def forward(
        self, 
        seq: torch.Tensor, 
        msa: torch.Tensor,
        seq_pad: Optional[torch.Tensor] = None,  # [B, L] True where padded
        msa_pad: Optional[torch.Tensor] = None   # [B, L] True where padded
    ) -> Tuple[torch.Tensor, torch.Tensor]: #TODO: Check for probability density in masked queries 
        for layer in self.layers:
            # seq attends to msa - normalize ALL inputs
            seq2, _ = layer["seq_attn"](
                layer["seq_norm"](seq),      # Normalize sequence queries
                layer["msa_norm"](msa),      # Normalize MSA keys (for stable attention weights)  
                layer["msa_norm"](msa),      # Normalize MSA values (for stable outputs)
                key_padding_mask=msa_pad
            )
            # msa attends to seq - normalize ALL inputs
            msa2, _ = layer["msa_attn"](
                layer["msa_norm"](msa),      # Normalize MSA queries
                layer["seq_norm"](seq),      # Normalize sequence keys (for stable attention weights)
                layer["seq_norm"](seq),      # Normalize sequence values (for stable outputs)
                key_padding_mask=seq_pad
            )
            seq, msa = seq + seq2, msa + msa2
            seq = layer["seq_ffn"](seq)
            msa = layer["msa_ffn"](msa)
        return seq, msa



############################################################
#  LightningModule - Optimized Implementation
############################################################

class ProteinLitModule(LightningModule):
    def __init__(
        self,
        task_type: str,                       # Task type: "mf", "bp", or "cc" - required
        d_model: int = 1152,                  # Base model dimension
        d_prot: int = 128,                    # Protein embedding dimension
        d_ankh: int = 1024,                   # Ankh3-Large embedding dimension
        d_msa: int = 768,                     # MSA embedding dimension
        n_cross_layers: int = 2,              # Cross-attention layers
        n_heads: int = 8,                     # Attention heads
        dropout: float = 0.1,                 # Dropout rate
        optimizer: Optional[Any] = None,      # Hydra optimizer config
        scheduler: Optional[Any] = None,      # Hydra scheduler config  
        warmup_ratio: float = 0.05,           # Warmup ratio for cosine schedule (5% of total training steps)
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for easy access
        self.task_type = task_type
        
        # Load MSA model - NO STREAM NEEDED
        import esm
        self.msa_model, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_model.eval().requires_grad_(False)
        # Learnable EOS token (inserted after MSA encoding as a register)
        self.eos_token = nn.Parameter(torch.zeros(1, 1, 1, d_msa))
        nn.init.normal_(self.eos_token, std=0.02)
        
        # Encoders
        self.seq_encoder = SeparateMHASequenceEncoder(
            d_model=d_model,
            d_prot=d_prot,  # Protein embedding dimension
            d_ankh=d_ankh,  # Ankh3-Large embedding dimension
            d_out=768,
            dropout=dropout,
            n_heads=n_heads
        )
        self.msa_encoder = MSAEncoder(
            d_msa=d_msa,
        )

        # Fusion / Cross-attention
        self.cross_attn = CrossModalAttention(
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
        """Setup hook handles MSA model device movement and compilation."""
        if stage == "fit":
            # MSA model compilation disabled by default for easier debugging
            print("🐛 MSA model compilation disabled for easier debugging and development")
            print(f"✅ ESM-MSA model loaded on {self.device}")

    # ---------------------------------------------------------------------
    #  Forward pass - optimized MSA computation
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optimized MSA processing and deferred EOS token handling.
        
        Key changes:
        - EOS token is excluded from MSA pooling by masking its position
        - Learnable EOS token is inserted after MSA encoding as a register
        - This ensures EOS doesn't interfere with attention/pooling operations
        """
        # Extract inputs - ESM-C embeddings are pre-computed
        seq_emb = batch["sequence_emb"]  # Pre-computed ESM-C: [B, L, d_model]
        prot_emb = batch["prot_emb"]     # Pre-computed protein: [B, L, d_prot]
        ankh_emb = batch["ankh_emb"]     # Pre-computed Ankh3-Large: [B, L, d_ankh]
        msa_tok_list = batch["msa_tok"]   # list[Tensor] – variable [N_seq_i, L_i]

        pad_mask = batch["pad_mask"]     # [B, L] - True where padded
        seq_pad    = batch["seq_pad_mask"]  # [B, N_max]

        # Compute MSA embeddings on-the-fly (sequential, per-sample)
        target_len = seq_emb.shape[1]  # CLS + residues + EOS (same as pad_mask width)
        msa_emb, msa_compute_time = self._compute_msa_embeddings(msa_tok_list, target_len)  # [B, N_seq_max, target_len, d_msa]
        
        # Expose timing so callbacks (e.g., BatchTimer) can log it **after** the forward pass.
        # Storing it on `self` ensures it is available in hooks such as `on_before_backward`.
        self._last_msa_compute_time = msa_compute_time

        # Keep original behaviour for potential external use
        batch["msa_compute_time"] = msa_compute_time

        # Mask EOS position for pooling-only (exclude from attention/pooling)
        eos_idx = batch["lengths"] + 1
        pooling_mask = pad_mask.clone()
        for i, pos in enumerate(eos_idx):
            pooling_mask[i, pos] = True

        # Encode each modality with regular padding masks (EOS included)
        seq_z = self.seq_encoder(seq_emb, prot_emb, ankh_emb, pad_mask=pad_mask, lengths=batch["lengths"])  # [B, L, 768]
        msa_z = self.msa_encoder(msa_emb, pad_mask=pooling_mask, seq_pad=seq_pad)  # [B, L, d]

        # Insert learnable EOS token after MSA encoding
        for i, pos in enumerate(eos_idx):
            msa_z[i, pos] = self.eos_token.squeeze()

        # Cross-modal attention with padding masks
        seq_z, msa_z = self.cross_attn(seq_z, msa_z, pad_mask, pad_mask)  # each [B, L, d]

        # Use masked mean pooling
        seq_pool = masked_mean_pooling(seq_z, pad_mask)  # [B, d]
        msa_pool = masked_mean_pooling(msa_z, pad_mask)  # [B, d]

        # Use AttentionFusion instead of concatenation
        fused = self.fusion(seq_pool, msa_pool)    # [B, d]
        
        logits = self.head(fused)                  # [B, C]
        
        return logits, batch["labels"]

    # ------------------------------------------------------------------
    #  Simple MSA embedding computation without streams
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _compute_msa_embeddings(
        self,
        msa_tok_list: List[torch.Tensor],
        target_len: int,
    ) -> Tuple[torch.Tensor, float]:
        """Compute MSA embeddings sequentially (one sample at a time).

        This avoids the huge memory spike that occurs when feeding the whole
        batch to ESM-MSA at once.  After all representations are computed we
        zero-pad them to a common shape so downstream code can keep treating
        the result as a single 4-D tensor: [B, N_seq_max, L_max, d_msa].
        """
        import time
        start = time.perf_counter()

        reps: List[torch.Tensor] = []
        max_n_seq = 0
        max_L     = 0

        for tok in msa_tok_list:
            # Move single sample to device and add batch dim expected by ESM-MSA
            tok = tok.unsqueeze(0)  # [1, N_seq, L]

            rep = self.msa_model(tok, repr_layers=[12])["representations"][12]

            rep = rep.squeeze(0)      # [N_seq, L, d_msa]
            reps.append(rep)

            # Track largest shapes for padding later
            max_n_seq = max(max_n_seq, rep.shape[0])

        # Ensure horizontal padding matches desired target_len
        max_L = target_len

        # ---------------------------------------------------------
        # Pad each representation to [max_n_seq, max_L, d_msa]
        # ---------------------------------------------------------
        padded_reps: List[torch.Tensor] = []
        for rep in reps:
            n_seq, L, D = rep.shape
            if n_seq < max_n_seq or L < max_L:
                padded = rep.new_zeros(max_n_seq, max_L, D) #TODO: CONSIDER PADDING TOKEN
                padded[:n_seq, :L] = rep
                rep = padded
            padded_reps.append(rep)

        msa_emb = torch.stack(padded_reps, dim=0)  # [B, N_seq_max, L_max, d_msa]

        return msa_emb, time.perf_counter() - start



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
       
    def configure_optimizers(self):
        if self.hparams.optimizer is None:
            raise ValueError("Optimizer must be provided in hparams")

        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = self.hparams.optimizer(params=trainable_params)

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

