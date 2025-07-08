from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


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


############################################################
#  Canonical Evoformer-Lite Implementation
############################################################

class _RowSelfAttention(nn.Module):
    """Self-attention along the **residue** dimension for every MSA row."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]):
        B, N, L, D = x.shape
        x_flat = x.reshape(B * N, L, D)
        pad_flat = None if pad_mask is None else pad_mask.unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        y, _ = self.attn(self.norm(x_flat), self.norm(x_flat), self.norm(x_flat), key_padding_mask=pad_flat)
        return x + y.reshape(B, N, L, D)


class _ColSelfAttention(nn.Module):
    """Self-attention along the **sequence-row** dimension for every residue column."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor):
        B, N, L, D = x.shape
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * L, N, D)
        y, _ = self.attn(self.norm(x_t), self.norm(x_t), self.norm(x_t))
        y = y.view(B, L, N, D).permute(0, 2, 1, 3)
        return x + y


class _ResidueSelfAttention(nn.Module):
    """Self-attention + FFN for residue track, with padding mask."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = ResidualFeedForward(d_model, 4, dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]):
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), key_padding_mask=pad_mask)
        x = x + y
        return self.ffn(x)


class EvoformerLiteBlock(nn.Module):
    """Canonical Evoformer-Lite block (row-attn, col-attn, residue self-attn)"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.row = _RowSelfAttention(d_model, n_heads, dropout)
        self.col = _ColSelfAttention(d_model, n_heads, dropout)
        self.msa_transition = ResidualFeedForward(d_model, 4, dropout)
        self.res_attn = _ResidueSelfAttention(d_model, n_heads, dropout)

    def forward(
        self,
        msa: torch.Tensor,           # [B,N,L,d]
        residue: torch.Tensor,       # [B,L,d]
        pad_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msa = self.row(msa, pad_mask)
        msa = self.col(msa)
        msa = self.msa_transition(msa)
        # Pool & fuse
        residue = residue + msa.mean(dim=1)
        residue = self.res_attn(residue, pad_mask)
        return msa, residue


class EvoformerLiteMSASeq(nn.Module):
    """Canonical Evoformer-Lite stack with MSA and residue tracks."""

    def __init__(self, d_msa: int, d_model: int, n_blocks: int, n_heads: int, dropout: float):
        super().__init__()
        self.msa_proj = nn.Linear(d_msa, d_model, bias=False)
        self.blocks = nn.ModuleList([EvoformerLiteBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, msa: torch.Tensor, residue: torch.Tensor, pad_mask: Optional[torch.Tensor]):
        msa = self.msa_proj(msa)
        for blk in self.blocks:
            msa, residue = blk(msa, residue, pad_mask)
        return residue


############################################################
#  LightningModule - Updated with Canonical Evoformer-Lite
############################################################

class ProteinLitModule(LightningModule):
    def __init__(
        self,
        task_type: str,                       # Task type: "mf", "bp", or "cc" - required
        d_model: int = 1152,                  # Base model dimension
        d_msa: int = 768,                     # MSA embedding dimension
        n_blocks: int = 4,                    # Number of Evoformer-Lite blocks
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
        
        # Load MSA model
        import esm
        self.msa_model, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_model.eval().requires_grad_(False)
        
        # Learnable EOS token to insert after residues
        self.eos_token = nn.Parameter(torch.zeros(1, 1, 1, d_msa))
        nn.init.normal_(self.eos_token, std=0.02)
        
        # Canonical Evoformer-Lite stack
        self.evo_stack = EvoformerLiteMSASeq(d_msa, d_model, n_blocks, n_heads, dropout)
        
        # Create classifier head
        self.head = MLPHead(d_model, get_num_classes_for_task(task_type), dropout)

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
    #  Forward pass with EOS token scatter
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with canonical Evoformer-Lite and EOS token scatter."""
        seq_emb = batch["sequence_emb"]          # [B,L+2,d_model]
        pad_mask = batch["pad_mask"]             # [B,L+2]
        lengths = batch["lengths"]               # [B] – residues (no CLS/EOS)
        msa_tok = batch["msa_tok"]               # list[Tensor]
        
        # Compute MSA embeddings (without EOS tokens)
        target_len = seq_emb.shape[1]  # CLS + residues + EOS (same as pad_mask width)
        msa_emb, msa_compute_time = self._compute_msa(msa_tok, target_len)  # [B,N,L+2,d_msa]
        
        # Insert EOS tokens using exact scatter procedure (outside inference mode)
        msa_emb = self.insert_eos_token(msa_emb, lengths)
        
        # Expose timing for callbacks
        self._last_msa_compute_time = msa_compute_time
        batch["msa_compute_time"] = msa_compute_time
        
        # Canonical Evoformer-Lite stack
        residue = self.evo_stack(msa_emb, seq_emb, pad_mask)
        
        # Masked mean-pool over length
        valid = (~pad_mask).sum(dim=1, keepdim=True)
        residue = residue.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = residue.sum(dim=1) / valid
        
        # Classification
        logits = self.head(pooled)
        
        return logits, batch["labels"]

    # ------------------------------------------------------------------
    #  EOS token insertion using exact scatter procedure
    # ------------------------------------------------------------------
    def insert_eos_token(self, msa: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Insert EOS tokens using exact scatter procedure from protein_module.py"""
        # Clone MSA since we can't modify in-place during inference
        msa = msa.clone()
        
        # ------------------------------------------------------------
        # Insert EOS (learnable) at position L+1  — truly in-place
        # ------------------------------------------------------------
        B, N_seq, L_pad, D = msa.shape
        # 0) prepare a (B × N_seq × 1 × D) tensor full of your EOS token
        eos_tok = self.eos_token.view(1, 1, 1, D).expand(B, N_seq, 1, D)    # (B, N_seq, 1, D)

        # 1) build the same shape index tensor for the "length+1" position
        idx = (lengths + 1).view(B, 1, 1, 1).expand(B, N_seq, 1, D)                # (B, N_seq, 1, D)

        # 2) scatter the EOS token into the msa along dim-2
        msa.scatter_(2, idx, eos_tok) 
        
        return msa

    # ------------------------------------------------------------------
    #  Simple MSA embedding computation without streams
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _compute_msa(self, msa_tok_list: List[torch.Tensor], target_len: int) -> Tuple[torch.Tensor, float]:
        """Compute MSA embeddings."""
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


