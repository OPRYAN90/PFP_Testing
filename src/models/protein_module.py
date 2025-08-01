from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

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


class RowDropout(nn.Module):
    """
    Drops *entire rows* in an MSA tensor with probability `p_row`.

    Inputs
    ------
    x       : Tensor[B, N_seq, L_pad, D]
    seq_pad : BoolTensor[B, N_seq]   – True where the row is padding

    Outputs
    -------
    x_out   : Tensor           – rows zeroed out where dropped
    seq_pad : BoolTensor       – updated to mark dropped rows as padding
    """
    def __init__(self, p_row: float = 0.1):
        super().__init__()
        self.p_row = float(p_row)

    def forward(self, x: torch.Tensor, seq_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.p_row <= 0.0:
            return x, seq_pad

        B, N, *_ = x.shape
        device    = x.device
        # Bernoulli mask – True means "keep"
        keep = (torch.rand(B, N, device=device) > self.p_row) & (~seq_pad)

        # Guarantee query sequence (first row) is always kept
        keep[:, 0] = True

        # broadcast to [B, N, 1, 1]
        keep_f = keep.unsqueeze(-1).unsqueeze(-1).float()
        x = x * keep_f

        seq_pad = seq_pad | (~keep)          # treat dropped rows as padding
        return x, seq_pad


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

class SequenceEncoder(nn.Module):
    """Encode pre-computed ESM-C embeddings with additional transformer layers.

    Input: Pre-computed ESM-C embeddings with shape [B, L, d_model] where:
    - B: batch size  
    - L: sequence length
    - d_model: ESM-C embedding dimension
    
    The encoder applies additional transformer layers on top of ESM-C features
    for task-specific fine-tuning.
    """

    def __init__(
        self,
        d_model: int = 1152,  # ESM-C embedding dimension
        n_layers: int = 2,   # Additional transformer layers
        n_heads: int = 8,    # Attention heads
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # ➊ optional embedding-level dropout
        self.embed_drop = nn.Dropout(dropout*0.75) 

        # # Additional transformer layers on top of ESM-C
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=n_heads,
        #     dim_feedforward=d_model * 4,
        #     dropout=dropout,
        #     batch_first=True,
        #     norm_first=True
        # )
        # self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, 768)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pre-computed ESM-C embeddings.
        
        Args:
            x: Pre-computed ESM-C embeddings [B, L, d_model]
            pad_mask: [B, L] True where padded
            
        Returns:
            Enhanced sequence representations [B, L, d_model]
        """
        x = self.embed_drop(x)                 # ➋ new dropout point
        # x = self.tr(x, src_key_padding_mask=pad_mask)
        x = self.proj(x)
        return x  # [B, L, d_model]


class MSAEncoder(nn.Module):
    """
    Row‑attention pooling encoder for an MSA tensor that already contains
    CLS (col 0) and the EOS token inserted by ProteinLitModule.insert_eos_token.

    Inputs
    -------
    msa       : [B, N_seq, L_pad, d_msa]
    pad_mask  : [B, L_pad]     – True where residue position is padding
    seq_pad   : [B, N_seq_max] – True where *row* is padding (absent sequence)

    Output
    ------
    x         : [B, L_pad, d_msa] – condensed L×D representation
    """

    def __init__(
        self,
        d_msa: int = 768,
        n_heads: int = 4,
        dropout_attn: float = 0.1,
        dropout_ffn: float = 0.1,
    ):
        super().__init__()
        self.feat_drop = nn.Dropout(dropout_attn)
        self.row_attn = nn.MultiheadAttention(
            d_msa, n_heads, dropout=dropout_attn, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_msa)
        self.norm_msa = nn.LayerNorm(d_msa)
        
        # Add gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_msa * 2, d_msa),
            nn.Sigmoid()
        )
        
        self.ffn = ResidualFeedForward(d_msa, dropout=dropout_ffn)
        self.out_norm = nn.LayerNorm(d_msa)

    def forward(
        self,
        msa: torch.Tensor,      # [B, N, L, D]
        pad_mask: torch.Tensor, # [B, L]
        seq_pad: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:

        B, N, L, D = msa.shape
        msa = self.feat_drop(msa)
        # ── 1. Pick the first row (query sequence) as the per‑residue query ──────
        q = msa[:, 0]                          # [B, L, D]
        # ── 2. Reshape so each residue column is an independent “set” ────────────
        msa_flat = msa.permute(0, 2, 1, 3)     # [B, L, N, D]
        msa_flat = msa_flat.reshape(B*L, N, D) # [B·L, N, D]
        q_flat   = q.reshape(B*L, 1, D)        # queries: [B·L, 1, D]

        # ── 3. Build row‑padding mask for MultiheadAttention ─────────────────────
        # seq_pad : [B, N]  →  repeat for every residue column
        row_pad = seq_pad.unsqueeze(1).expand(-1, L, -1)        # [B, L, N]
        row_pad = row_pad.reshape(B*L, N)                       # [B·L, N]

        # ── 4. Row‑wise multi‑head attention ─────────────────────────────────────
        z, _ = self.row_attn(
            query=self.norm_q(q_flat),                 # [B·L, 1, D]
            key=self.norm_msa(msa_flat), value=self.norm_msa(msa_flat), # [B·L, N, D]
            key_padding_mask=row_pad      # mask padded rows
        )                                 # → [B·L, 1, D]
        z = z.reshape(B, L, D)            # [B, L, D]

        # ── 5. Apply gating between query and MSA-informed features ──────────────
        gate_input = torch.cat([q, z], dim=-1)  # [B, L, 2*D]
        gate = self.gate(gate_input)            # [B, L, D]
        z = gate * q + (1 - gate) * z           # Gated combination

        # ── 6. Residual + FFN (per residue) ──────────────────────────────────────
        z = self.ffn(z)                   # [B, L, D]

        # ── 7. Zero‑out padded residue positions (keep tensor shape) ────────────
        z = z.masked_fill(pad_mask.unsqueeze(-1), 0.0)  # [B, L, D]

        return z


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        d_msa: int = 768,                     # MSA embedding dimension
        n_seq_layers: int = 2,                # Sequence encoder layers
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
        # Learnable EOS token (shared across alignment rows)
        self.eos_token = nn.Parameter(torch.zeros(1, 1, 1, d_msa))
        nn.init.normal_(self.eos_token, std=0.02)
        
        # Encoders
        self.seq_encoder = SequenceEncoder(
            d_model=d_model,
            n_layers=n_seq_layers,
            n_heads=n_heads,
            dropout=dropout
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
        """Forward pass with all three improvements."""
        # Extract inputs - ESM-C embeddings are pre-computed
        seq_emb = batch["sequence_emb"]  # Pre-computed ESM-C: [B, L, d_model]
        msa_tok_list = batch["msa_tok"]   # list[Tensor] – variable [N_seq_i, L_i]

        pad_mask = batch["pad_mask"]     # [B, L] - True where padded
        seq_pad    = batch["seq_pad_mask"]  # [B, N_max]

        # Compute MSA embeddings on-the-fly (sequential, per-sample)
        target_len = seq_emb.shape[1]  # CLS + residues + EOS (same as pad_mask width)
        msa_emb, msa_compute_time = self._compute_msa_embeddings(msa_tok_list, target_len)  # [B, N_seq_max, target_len, d_msa]
        # Insert EOS token (learnable) before passing the embeddings to the encoders
        msa_emb = self.insert_eos_token(msa_emb, batch["lengths"])
        # Expose timing so callbacks (e.g., BatchTimer) can log it **after** the forward pass.
        # Storing it on `self` ensures it is available in hooks such as `on_before_backward`.
        self._last_msa_compute_time = msa_compute_time

        # Keep original behaviour for potential external use
        batch["msa_compute_time"] = msa_compute_time

        # derive an MSA-style mask  – identical to seq mask for per-residue padding
        msa_pad_mask = pad_mask    # shape [B, L]
        seq_pad_mask = pad_mask

        # Encode each modality with padding masks
        seq_z = self.seq_encoder(seq_emb, pad_mask=seq_pad_mask)  # [B, L, d]
        msa_z = self.msa_encoder(msa_emb, pad_mask=msa_pad_mask, seq_pad=seq_pad)  # [B, L, d]

        # Cross-modal attention with padding masks
        seq_z, msa_z = self.cross_attn(seq_z, msa_z, seq_pad_mask, msa_pad_mask)  # each [B, L, d]

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
    #  EOS token insertion (moved from MSAEncoder)
    # ------------------------------------------------------------------
    def insert_eos_token(self, msa: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scatter a learnable EOS token into the MSA embeddings.

        Args:
            msa: Tensor of shape [B, N_seq, L_pad, d_msa] without EOS tokens.
            lengths: Tensor [B] containing the true residue counts (excluding CLS/EOS).

        Returns:
            Tensor with EOS token inserted at index (length + 1) for every sequence row.
        """
        # Clone to avoid inadvertent in-place modifications further upstream
        msa = msa.clone()

        B, N_seq, L_pad, D = msa.shape

        # Expand the single learnable EOS vector to required shape
        eos_tok = self.eos_token.view(1, 1, 1, D).expand(B, N_seq, 1, D)

        # Build index tensor for the EOS position
        idx = (lengths + 1).view(B, 1, 1, 1).expand(B, N_seq, 1, D)

        # Scatter EOS tokens into the residue dimension
        msa.scatter_(2, idx, eos_tok)
        return msa

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

