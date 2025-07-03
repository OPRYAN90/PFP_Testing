from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
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

############################################################
#  Low-level building blocks
############################################################

class RowDropout(nn.Module):
    """
    Zero out entire MSA rows.  Probability decays to zero when the
    alignment has fewer than `min_keep` sequences.
    The query row (index 0) is always kept.
    """
    def __init__(self, p: float = 0.15, min_keep: int = 32):
        super().__init__()
        self.base_p   = p
        self.min_keep = min_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B, N_seq, _, _ = x.shape
        # linear annealing: p → 0 as N_seq ↓
        p_eff = self.base_p * min(1.0, N_seq / self.min_keep)
        if p_eff == 0:
            return x

        mask = torch.rand(B, N_seq, 1, 1, device=x.device) > p_eff
        mask[:, 0, :, :] = 1.0                     # keep query
        return x * mask                      # no rescaling



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

class RowwiseChannelDropout(nn.Module):
    """
    AlphaFold-style: drop embedding channels; mask shared across all rows.
    """
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # mask shape: [B, 1, 1, d_msa] (shared across N_seq & L)
        mask = torch.rand(x.size(0), 1, 1, x.size(3), device=x.device) > self.p
        return x * mask / (1.0 - self.p)  # WITH rescaling for gradient stability


class MLPHead(nn.Module):
    """Final classification head with layer norm and dropout."""
    
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1): 
        super().__init__()
        assert d_in == 1152, "d_in must be 1152"
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, d_out), #EXTREME WARNING NOTE: DIMS DEPEDNING ON TASK AND D_MSA
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        p_row: float = 0.15,    # dropout prob for individual MSA rows
        p_chan: float = 0.10,   # channel dropout prob 
        p_feat: float = 0.10,   # feature dropout after projection
    ):
        super().__init__()
        
        # 0) Learnable EOS token shared across rows
        self.eos_token = nn.Parameter(torch.zeros(1, 1, 1, d_msa))
        nn.init.normal_(self.eos_token, std=0.02)
        
        # 1) Row-level dropout (zeros individual sequences)
        self.row_dropout = RowDropout(p_row, min_keep=32)

        self.channel_dropout = RowwiseChannelDropout(p_chan)
                
    def forward(
        self,
        msa: torch.Tensor,   # [B, N_seq, L_pad, d_msa]  zero-padded
        lengths: torch.Tensor,   # [B]  true residue counts (no CLS/EOS)
        pad_mask: torch.Tensor,  # [B, L] True where padded
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # Insert EOS (learnable) at position L+1  — truly in-place
        # ------------------------------------------------------------
        eos_idx = lengths + 1                      # [B]

        B, N_seq, _, D = msa.shape
        # Build broadcastable index tensors
        batch_idx = torch.arange(B,  device=msa.device)[:, None, None]
        seq_idx   = torch.arange(N_seq, device=msa.device)[None, :, None]
        # eos_idx already has shape [B]; add singleton dims
        pos_idx   = eos_idx[:, None, None]
        if torch.compiler.is_compiling():  # cheaper than .clone()
            torch._dynamo.graph_break()   # forces eager mode for next operation
        msa[batch_idx, seq_idx, pos_idx, :] = self.eos_token

        # ------------------------------------------------------------
        # 1. Row dropout  ─── 2. Channel dropout  ─── 3-5. Rest
        # ------------------------------------------------------------
        msa     = self.row_dropout(msa)
        x     = self.channel_dropout(msa)
        
        return x


############################################################
#  LightningModule - Adapted for Lightning-Hydra Setup
############################################################

class ProteinLitModule(LightningModule):
    """LightningModule for multi-modal protein function prediction.

    Adapted for Lightning-Hydra template with dynamic num_classes and 
    Hydra-compatible optimizer/scheduler configuration.
    
    """

    def __init__(
        self,
        task_type: str = None,        # Task type: "mf", "bp", or "cc"
        d_model: int = 1152,           # Base model dimension
        d_msa: int = 768,             # MSA embedding dimension
        n_seq_layers: int = 2,        # Sequence encoder layers
        # n_msa_layers: int = 2,        # MSA encoder layers  
        n_cross_layers: int = 2,      # Cross-attention layers
        n_heads: int = 8,             # Attention heads
        dropout: float = 0.1,         # Dropout rate
        # MSA Encoder dropout parameters
        p_row: float = 0.15,          # Row dropout probability (individual MSA sequences)
        p_chan: float = 0.15,         # Channel dropout probability (AlphaFold-style)
        p_feat: float = 0.10,         # Feature dropout after MSA projection
        optimizer: torch.optim.Optimizer = None,  # Hydra optimizer config
        scheduler = None,  # Hydra scheduler config  
        debugging: bool = True,       # Default to debugging mode (disables compilation)
        warmup_ratio: float = 0.05,   # Warmup ratio for cosine schedule (5% of total training steps)
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

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
            p_row=p_row,
            p_chan=p_chan,
            p_feat=p_feat
        )

        # Fusion / Cross-attention
        self.cross_attn = CrossModalAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout
        )

        # Readout: concatenate residue-wise features (mean-pool) -> linear projection
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        
        # Create classifier head in __init__ now that we have num_classes
        self.head = MLPHead(d_model, get_num_classes_for_task(self.hparams.task_type), dropout)

        # For multi-label protein prediction, we use standard BCE loss
        self.loss_fn = nn.BCEWithLogitsLoss() #TODO: Consider weighted loss by adding InfoNCE loss funciton 

        # Loss tracking (these don't depend on num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # Storage for epoch-wise sweep for CAFA metrics
        self._val_logits = []
        self._val_labels = []
        self._test_logits = []
        self._test_labels = []

    def setup(self, stage: str) -> None:
        """Setup hook now only handles compilation and datamodule validation."""
        # Validate that datamodule num_classes matches model num_classes (if available)
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'num_classes'):
            datamodule_classes = self.trainer.datamodule.num_classes
            model_classes = get_num_classes_for_task(self.hparams.task_type)
            
            if datamodule_classes != model_classes:
                raise ValueError(
                    f"Mismatch between model num_classes ({model_classes}) "
                    f"and datamodule num_classes ({datamodule_classes})"
                )
            
            print(f"✅ Validated model-datamodule compatibility: {model_classes} classes")
        else:
            print("No datamodule found, skipping validation")
        
        # Compilation logic: Compile when NOT debugging
        if stage == "fit" and not self.hparams.debugging:
            print("🚀  Production mode: compiling heavy encoders")
            self.seq_encoder = torch.compile(self.seq_encoder)
            self.msa_encoder = torch.compile(self.msa_encoder)
            self.cross_attn = torch.compile(self.cross_attn)
        elif stage == "fit":
            print("🐛 Debugging mode: Compilation disabled for easier debugging")

    # ---------------------------------------------------------------------
    #  Forward pass - adapted for your datamodule's batch format
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass adapted for ESM and MSA modalities only."""
        # Extract inputs - ESM-C embeddings are pre-computed
        seq_emb = batch["sequence_emb"]  # Pre-computed ESM-C: [B, L, d_model]
        msa_emb = batch["msa_emb"]       # MSA embeddings: [B, N_seq, L, d_msa] 
        pad_mask = batch["pad_mask"]     # [B, L] - True where padded

        # derive an MSA‐style mask
        # pad_mask: True at pad → key_padding_mask expects True==mask
        msa_pad_mask = pad_mask    # keep shape [B, L]
        seq_pad_mask = pad_mask

        # Encode each modality with padding masks
        seq_z = self.seq_encoder(seq_emb, src_key_padding_mask=pad_mask)  # [B, L, d]
        msa_z = self.msa_encoder(msa_emb, batch["lengths"], pad_mask=pad_mask)  # [B, L, d]

        # Cross-modal attention with padding masks
        seq_z, msa_z = self.cross_attn(seq_z, msa_z, seq_pad_mask, msa_pad_mask)  # each [B, L, d]

        # Do masked pooling instead of naive .mean()
        # compute length of each sequence (non-padding)
        valid_counts = (~pad_mask).sum(dim=1, keepdim=True).float()  # [B, 1]

        # zero out pad positions and do masked average
        seq_z_masked = seq_z.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        seq_pool = seq_z_masked.sum(dim=1) / valid_counts  # [B, d]
        msa_z_masked = msa_z.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        msa_pool = msa_z_masked.sum(dim=1) / valid_counts  # [B, d]

        fused = torch.cat([seq_pool, msa_pool], dim=-1)  # [B, 2d]
        fused = self.fusion_proj(fused)                   # [B, d]
        logits = self.head(fused)                         # [B, C]
        
        return logits, batch["labels"].float()

    # ------------------------------------------------------------------
    #  Lightning hooks - compatible with Lightning-Hydra template
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.train_loss(loss)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.val_loss(loss)

        # Store for epoch-wise CAFA metrics computation
        self._val_logits.append(logits.detach().cpu())   # keep raw, no sigmoid
        self._val_labels.append(labels.detach().cpu())
        
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.test_loss(loss)
 
        # Store for epoch-wise CAFA metrics computation
        self._test_logits.append(logits.detach().cpu())   # keep raw, no sigmoid
        self._test_labels.append(labels.detach().cpu())

    def _heal_metrics(self, logits, labels):
        probs = torch.sigmoid(logits).numpy()
        labels = labels.numpy().astype(int)
        goterms = list(self.trainer.datamodule._go_dicts[self.hparams.task_type].keys())

        # Parent-child propagation
        if self.hparams.task_type == "ec": #NOTE: If ec is enabled update this
            print("EC task type detected; ENSURE UPDATE")
            probs = propagate_ec_preds(probs.copy(), goterms)
        else:
            probs = propagate_go_preds(probs.copy(), goterms)

        macro, micro = function_centric_aupr(labels, probs)
        fmax, _      = cafa_fmax(labels, probs, goterms, self.hparams.task_type)
        s_min        = smin(labels, probs, self.trainer.datamodule.ic_vector.numpy())

        return macro, micro, fmax, s_min

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits)
        labels = torch.cat(self._val_labels)
        macro, micro, fmax, s_min = self._heal_metrics(logits, labels)

        self.log_dict({
            "val/loss": self.val_loss,
            "val/AUPR_macro": macro,
            "val/AUPR_micro": micro,
            "val/Fmax": fmax,
            "val/Smin": s_min,
        }, prog_bar=True, sync_dist=True)

        self.val_loss.reset(); self._val_logits.clear(); self._val_labels.clear()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset validation metrics to avoid storing results from sanity checks
        self.val_loss.reset()
        self._val_logits.clear()
        self._val_labels.clear()
       
    def configure_optimizers(self):
            optimizer = self.hparams.optimizer(params=self.parameters())

            # ---- 100 % bullet-proof step count ---------------------------
            total_steps = self.trainer.estimated_stepping_batches  # ← 1-liner

            # 5 % warm-up, but expose it as Hydra-overridable hyper-param
            warmup_steps = int(self.hparams.warmup_ratio * total_steps)

            if self.hparams.scheduler is not None:
                # `get_cosine_schedule_with_warmup` signature:
                # (optimizer, num_warmup_steps, num_training_steps, …)  :contentReference[oaicite:1]{index=1}
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
                        "interval": "step",      # step every optimizer.step()
                        "frequency": 1,
                    },
                }

            return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.train_loss.reset()  # Explicit reset for consistency

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        logits = torch.cat(self._test_logits)
        labels = torch.cat(self._test_labels)
        macro, micro, fmax, s_min = self._heal_metrics(logits, labels)

        self.log_dict({
            "test/loss": self.test_loss,
            "test/AUPR_macro": macro,
            "test/AUPR_micro": micro,
            "test/Fmax": fmax,
            "test/Smin": s_min,
        }, prog_bar=True, sync_dist=True)

        self.test_loss.reset(); self._test_logits.clear(); self._test_labels.clear()


