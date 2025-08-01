from typing import Any, Dict, Optional, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # assert d_in == 1152, "d_in must be 1152"
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 602),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(602, d_out), #EXTREME WARNING NOTE: DIMS DEPEDNING ON TASK AND D_MSA
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class AFF2D(nn.Module):
    """
    Adaptive‑Frequency‑Filter token mixer (channel–last).
    Args
    ----
    hidden_size : C  – number of channels coming from the previous block
    num_groups  : G  – number of groups for the group‑wise 1×1 “mask MLP”
    mask_hidden : r  – expansion ratio for the hidden layer inside the mask MLP
    Forward
    -------
    x : (B, H*W, C)     – flattened tokens
    spatial_size : (H, W)
    returns : (B, H*W, C)   – filtered tokens (residual added)
    """
    def __init__(self,
                 hidden_size: int,
                 num_groups: int = 8,
                 mask_hidden: int = 4):
        super().__init__()
        if hidden_size % num_groups:
            raise ValueError('hidden_size must be divisible by num_groups')

        self.hidden_size   = hidden_size             # C
        self.num_groups    = num_groups              # G
        self.freq_channels = 2 * hidden_size         # real ⊕ imag ⇒ 2C

        # Group‑wise 1×1‑MLP that predicts the complex mask  M  in Eq.(4):contentReference[oaicite:0]{index=0}
        self.mask_fc1 = nn.Conv2d(
            in_channels  = self.freq_channels,
            out_channels = mask_hidden * self.freq_channels,
            kernel_size  = 1,
            groups       = num_groups,
            bias         = True,
        )
        self.act = nn.ReLU(inplace=True)
        self.mask_fc2 = nn.Conv2d(
            in_channels  = mask_hidden * self.freq_channels,
            out_channels = self.freq_channels,
            kernel_size  = 1,
            groups       = num_groups,
            bias         = True,
        )

    # keep FFTs in fp32 for numerical stability (paper uses the same precaution)
    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor, spatial_size: tuple[int, int]):
        B, HW, C = x.shape
        H, W     = spatial_size
        if HW != H * W or C != self.hidden_size:
            raise ValueError('`spatial_size` inconsistent with input')

        # -> (B, C, H, W)
        x_img = x.transpose(1, 2).reshape(B, C, H, W)

        # 2‑D real FFT  (orthonormal scaling keeps energy unchanged)
        Xf = torch.fft.rfft2(x_img, dim=(-2, -1), norm='ortho')

        # split into real/imag and stack along channel dim  → (B, 2C, H, W//2+1)
        Xf_cat = torch.cat([Xf.real, Xf.imag], dim=1)

        # learn instance‑adaptive mask  M(F(X))  (same shape as Xf_cat)
        M = self.mask_fc2(self.act(self.mask_fc1(Xf_cat)))

        # complex multiplication — Eq.(4) Hadamard product in frequency domain
        M_real, M_imag = M.chunk(2, dim=1)
        Yf = torch.complex(
            M_real * Xf.real - M_imag * Xf.imag,
            M_real * Xf.imag + M_imag * Xf.real,
        )

        # inverse FFT  (B, C, H, W)  — gives F⁻¹[M⊙F(X)]
        y_img = torch.fft.irfft2(Yf, s=(H, W), dim=(-2, -1), norm='ortho')

        # reshape back & add residual
        y = y_img.reshape(B, C, HW).transpose(1, 2)
        return y


class AFFMix2D(nn.Module):     # mirrors your AFNOMix2D
    def __init__(self, d_model, num_groups=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.aff  = AFF2D(hidden_size=d_model, num_groups=num_groups)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        x = x.masked_fill(pad.unsqueeze(-1), 0.)
        B,N,L,D = x.shape
        y = self.norm(x).reshape(B, N*L, D)          # [B, N*L, D]
        y = self.aff(y, spatial_size=(N, L))
        y = y.reshape(B, N, L, D).masked_fill(pad.unsqueeze(-1), 0.)
        return x + self.drop(y)

class _ResidueSelfAttention(nn.Module):
    """Self-attention + FFN for residue track, with padding mask."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = ResidualFeedForward(d_model, 4, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = x + attn_out
        return self.ffn(x)

# -----------------------------------------------
# EvoformerLite Block
# -----------------------------------------------
class EvoformerLiteBlock(nn.Module):
    """Canonical Evoformer-Lite block with ConvMix2D"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float, *_):
        super().__init__()
        
        self.mix = AFFMix2D(d_model)   # NEW
        self.msa_transition = ResidualFeedForward(d_model, 4, dropout)

        # --- MSA → residue fuse & residue self-attn remain unchanged ---
        self.seq_norm      = nn.LayerNorm(d_model)
        self.seq_out_proj  = nn.Linear(d_model, d_model, bias=True)
        self.seq_gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.seq_out_proj.weight);  nn.init.zeros_(self.seq_out_proj.bias)
        nn.init.zeros_(self.seq_gate_proj.weight); nn.init.constant_(self.seq_gate_proj.bias, 1.0)
        self.seq_dropout   = nn.Dropout(dropout)
        self.res_attn      = _ResidueSelfAttention(d_model, n_heads, dropout)

    def forward(self, msa, residue, pad_mask, seq_pad_mask):
        # Build 3-D padding mask for MSA tokens: True ⇢ PAD
        #   seq_pad_mask : [B, N]
        #   pad_mask     : [B, L]
        pad = seq_pad_mask.unsqueeze(-1) | pad_mask.unsqueeze(1)  # [B,N,L]

        msa = self.mix(msa, pad)          # ① Conv mixing with mask
        msa = self.msa_transition(msa)    # ② Position-wise FFN
        # ③ MSA → residue projection (unchanged)
        seq_mask   = (~seq_pad_mask).unsqueeze(-1).unsqueeze(-1).type_as(msa)
        seq_valid  = seq_mask.sum(dim=1)
        seq_mean   = (msa * seq_mask).sum(dim=1) / seq_valid     # [B, L, D]
        seq_mean   = self.seq_norm(seq_mean)
        update     = self.seq_out_proj(seq_mean)
        gate       = torch.sigmoid(self.seq_gate_proj(seq_mean))
        residue    = residue + self.seq_dropout(gate * update)
        # ④ Residue self-attention (unchanged)
        residue    = self.res_attn(residue, pad_mask)
        return msa, residue


class EvoformerLiteMSASeq(nn.Module):
    """Canonical Evoformer-Lite stack with MSA and residue tracks."""

    def __init__(self, d_msa: int, d_model: int, n_blocks: int, n_heads: int, dropout: float, in_dropout: float):
        super().__init__()

        self.msa_proj = nn.Linear(d_msa, d_model, bias=False)
        self.esm_proj = nn.Linear(1152, d_model, bias=False)
        self.in_drop = nn.Dropout(in_dropout)

        # Evoformer-Lite tower (updated to use ConvMix2D)
        self.blocks = nn.ModuleList([
            EvoformerLiteBlock(d_model, n_heads, dropout) 
            for _ in range(n_blocks)
        ])

    def forward(self, msa, residue, pad_mask, seq_pad_mask):
        # MSA path
        msa = self.in_drop(self.msa_proj(msa))
        # Residue path (ESM-C already has correct dim; leave untouched)
        residue = self.in_drop(self.esm_proj(residue))

        for blk in self.blocks:
            msa, residue = blk(msa, residue, pad_mask, seq_pad_mask)
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
        in_dropout: float = 0.1,              # Input dropout rate
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
        
        # Canonical Evoformer-Lite stack with ConvMix2D
        self.evo_stack = EvoformerLiteMSASeq(d_msa, d_model, n_blocks, n_heads, dropout, in_dropout)
        
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
        pad_mask   = batch["pad_mask"]          # [B, L+2]  (True → PAD)
        seq_pad    = batch["seq_pad_mask"]  # [B, N_max]
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
        
        # sanity: ensure N_seq dims still agree after ESM-MSA padding
        assert msa_emb.shape[1] == seq_pad.shape[1], "N_seq mismatch after _compute_msa"

        # Canonical Evoformer-Lite stack with ConvMix2D and sequence padding mask
        residue = self.evo_stack(msa_emb, seq_emb, pad_mask, seq_pad)
        
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
        eos_tok = self.eos_token.reshape(1, 1, 1, D).expand(B, N_seq, 1, D) # (B, N_seq, 1, D)

        # 1) build the same shape index tensor for the "length+1" position
        idx = (lengths + 1).reshape(B, 1, 1, 1).expand(B, N_seq, 1, D)             # (B, N_seq, 1, D)

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
                padded = rep.new_zeros(max_n_seq, max_L, D) 
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

