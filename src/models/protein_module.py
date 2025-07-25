from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Optional FlashAttention toggle – does nothing on unsupported systems
# -----------------------------------------------------------------------------

from utils.flash_control import maybe_enable_flash_attention  
maybe_enable_flash_attention(True)


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
    """Encoder for Multiple Sequence Alignment (MSA) embeddings.
    Expected input: msa [B, N_seq, L_cls+res, d_msa] (already includes CLS at col 0)
    - B: batch size
    - N_seq: alignment depth (≈256) 
    - L_cls+res: CLS token + residue length
    - d_msa: embedding dim from MSA-Transformer (≈768)
    """
    
    def __init__(
        self,
        d_msa: int = 768,
        d_model: int = 1152,
        p_chan: float = 0.10,   # channel dropout prob 
        p_feat: float = 0.10,   # feature dropout after projection
    ):
        super().__init__()
        
        
        # 3) Conservation scoring head
        self.conservation_head = nn.Linear(d_msa, 1, bias=False)
        
 
        self.post_ffn   = ResidualFeedForward(d_msa, expansion=4, dropout=p_feat)
        self.norm = nn.LayerNorm(d_msa)
    def forward(
        self,
        msa: torch.Tensor,   # [B, N_seq, L_pad, d_msa]  zero-padded
        lengths: torch.Tensor,   # [B]  true residue counts (no CLS/EOS)
        pad_mask: torch.Tensor,  # [B, L] True where padded
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # Insert EOS (learnable) at position L+1  — truly in-place
        # ------------------------------------------------------------


        scores  = self.conservation_head(msa).squeeze(-1)     # [B, N_seq, L_pad]
        
        # Mask out padded positions before softmax
        # pad_mask: [B, L] → expand to [B, N_seq, L]
        mask = pad_mask.unsqueeze(1).expand(-1, msa.size(1), -1)      
        scores   = scores.masked_fill(mask, -1e4)                  # never get prob-mass
        
        # Detect rows that are entirely padding (vertical padding from collate_fn)
        # If first position (CLS token) of a row has all zeros, entire row is padding
        row_padding_mask = (msa[:, :, 0, :].abs().sum(-1) == 0)    # [B, N_seq] - True if row is padding
        row_padding_mask = row_padding_mask.unsqueeze(-1).expand(-1, -1, msa.size(2))  # [B, N_seq, L_pad]
        
        # Combine position-wise dropout mask with row padding mask
        mask_logits = row_padding_mask & (~mask)                          # only real residues
        scores = scores.masked_fill(mask_logits, -1e4)

        weights = torch.softmax(scores, dim=1)                # over N_seq
        weights = weights.masked_fill(mask, 0.0)       # ← explicit zero-out after softmax
        pooled  = (msa * weights.unsqueeze(-1)).sum(dim=1)    # [B, L_pad, d_msa]
        x = self.norm(pooled)
        x = self.post_ffn(x)
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
        # MSA Encoder dropout parameters
        p_chan: float = 0.15,                 # Channel dropout probability (AlphaFold-style)
        p_feat: float = 0.10,                 # Feature dropout after MSA projection
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
        self.channel_dropout = nn.Dropout(p_chan)
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
            d_model=d_model,
            p_chan=p_chan,
            p_feat=p_feat
        )

        # Fusion / Cross-attention
        self.cross_attn = CrossModalAttention(
            d_model=768,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout
        )

        # Readout: concatenate residue-wise features (mean-pool) -> linear projection
        self.fusion_proj = nn.Linear(768 * 2, 768)
        
        # Create classifier head
        self.head = MLPHead(768, get_num_classes_for_task(task_type), dropout)
        

        self.msa_encoder_2 = MSAEncoder(
            d_msa=d_msa,
            d_model=d_model,
            p_chan=p_chan,
            p_feat=p_feat
        )
        
        self.cross_attn_2 = CrossModalAttention(
            d_model=768,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout
        )
        
        self.cross_attn_3 = CrossModalAttention(
            d_model=768,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout
        )

        self.msa_encoder_3 = MSAEncoder(
            d_msa=d_msa,
            d_model=d_model,
            p_chan=p_chan,
            p_feat=p_feat
        )
        
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
        """Forward pass adapted for ESM and MSA modalities only."""
        # Extract inputs - ESM-C embeddings are pre-computed
        seq_emb = batch["sequence_emb"]  # Pre-computed ESM-C: [B, L, d_model]
        msa_tok_list = batch["msa_tok"]   # list[Tensor] – variable [N_seq_i, L_i]

        pad_mask = batch["pad_mask"]     # [B, L] - True where padded

        # Compute MSA embeddings on-the-fly (sequential, per-sample)
        target_len = seq_emb.shape[1]  # CLS + residues + EOS (same as pad_mask width)
        msa_emb, msa_compute_time = self._compute_msa_embeddings(msa_tok_list, target_len)  # [B, N_seq_max, target_len, d_msa]
        msa = msa_emb.clone()
        B, N_seq, L_pad, D = msa_emb.shape
        eos_tok = self.eos_token.view(1, 1, 1, D).expand(B, N_seq, 1, D)    # (B, N_seq, 1, D)

        # 1) build the same shape index tensor for the "length+1" position
        idx = (batch["lengths"] + 1).view(B, 1, 1, 1).expand(B, N_seq, 1, D)                # (B, N_seq, 1, D)

        # 2) scatter the EOS token into the msa along dim-2
        msa.scatter_(2, idx, eos_tok) 

        msa     = self.channel_dropout(msa)
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
        msa_z = self.msa_encoder(msa, batch["lengths"], pad_mask=msa_pad_mask)  # [B, L, d]

        # Cross-modal attention with padding masks
        seq_z, msa_z = self.cross_attn(seq_z, msa_z, seq_pad_mask, msa_pad_mask)  # each [B, L, d]
            
        msa_update = self.msa_encoder_2(msa, batch["lengths"], pad_mask=msa_pad_mask)
        msa_update = msa_update + msa_z
        seq_update, msa_update = self.cross_attn_2(seq_z, msa_update, seq_pad_mask, msa_pad_mask)
        
        msa_update_2 = self.msa_encoder_3(msa, batch["lengths"], pad_mask=msa_pad_mask)
        msa_update_2 = msa_update + msa_update_2
        seq_update, msa_update_2 = self.cross_attn_3(seq_update, msa_update_2, seq_pad_mask, msa_pad_mask)
        
        seq_z = seq_update
        msa_z = msa_update_2
        
        
        # Do masked pooling instead of naive .mean()
        # compute length of each sequence (non-padding)
        valid_counts = (~pad_mask).sum(dim=1, keepdim=True) # [B, 1]

        # zero out pad positions and do masked average
        seq_z_masked = seq_z.masked_fill(pad_mask.unsqueeze(-1), 0.0)  
        seq_pool = seq_z_masked.sum(dim=1) / valid_counts  # [B, d]
        msa_z_masked = msa_z.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        msa_pool = msa_z_masked.sum(dim=1) / valid_counts  # [B, d]

        fused = torch.cat([seq_pool, msa_pool], dim=-1)  # [B, 2d]
        fused = self.fusion_proj(fused)                   # [B, d]
        logits = self.head(fused)                         # [B, C]
        
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


