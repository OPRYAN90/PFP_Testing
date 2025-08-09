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
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

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
    def __init__(self, d_model=768, p_drop=0.1, favor_seq_bias=0.0):
        super().__init__()
        self.pre = nn.LayerNorm(2 * d_model)
        self.dropout = nn.Dropout(p_drop)
        self.gate = nn.Linear(2 * d_model, 2)
        # Optional: bias > 0 favors seq at init; 0.0 is neutral (50/50)
        nn.init.constant_(self.gate.bias, favor_seq_bias)

        # (Nice default) zero-init weights so bias controls the initial mix
        nn.init.zeros_(self.gate.weight)

    def forward(self, seq_feat, msa_feat):
        h = torch.cat([seq_feat, msa_feat], dim=-1)
        logits = self.gate(self.dropout(self.pre(h)))   # [B, 2]
        w = torch.softmax(logits, dim=-1)               # [B, 2]
        return w[..., 0:1] * seq_feat + w[..., 1:2] * msa_feat



############################################################
#  Encoders for the three modalities
############################################################

class CrossAttentionFusion(nn.Module):
    """Cross-attention between all three modalities with softmax gating"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention for each modality pair
        self.esm_to_others = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prot_to_others = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ankh_to_others = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # NEW: per-token gating over {esm, prot, ankh}
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Dropout(0.1),                # (optional) on inputs to the gate
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),                # on hidden activations
            nn.Linear(d_model // 2, 3)      # logits -> softmax outside
        )
        
        # Final combination layer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, esm, prot, ankh, pad_mask=None):
        # Create key-value pairs from other modalities
        others_esm = torch.cat([prot, ankh], dim=1)  # [B, 2L, d]
        others_prot = torch.cat([esm, ankh], dim=1)  # [B, 2L, d]
        others_ankh = torch.cat([esm, prot], dim=1)  # [B, 2L, d]
        
        # Create proper padding masks for concatenated sequences
        if pad_mask is not None:
            # For others_esm (prot + ankh): duplicate pad_mask twice
            others_esm_mask = torch.cat([pad_mask, pad_mask], dim=1)  # [B, 2L]
            # For others_prot (esm + ankh): duplicate pad_mask twice  
            others_prot_mask = torch.cat([pad_mask, pad_mask], dim=1)  # [B, 2L]
            # For others_ankh (esm + prot): duplicate pad_mask twice
            others_ankh_mask = torch.cat([pad_mask, pad_mask], dim=1)  # [B, 2L]
        else:
            others_esm_mask = others_prot_mask = others_ankh_mask = None
        
        # Apply padding mask to query tensors before cross-attention
        if pad_mask is not None:
            # Zero out padded positions in query tensors
            esm = esm.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            prot = prot.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            ankh = ankh.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        
        # Cross-attention with correct padding masks
        esm_attended, _ = self.esm_to_others(esm, others_esm, others_esm, key_padding_mask=others_esm_mask)
        prot_attended, _ = self.prot_to_others(prot, others_prot, others_prot, key_padding_mask=others_prot_mask)
        ankh_attended, _ = self.ankh_to_others(ankh, others_ankh, others_ankh, key_padding_mask=others_ankh_mask)
        
        # NEW: convex (softmax) gating per token over the three streams
        gate_in = torch.cat([esm_attended, prot_attended, ankh_attended], dim=-1)   # [B, L, 3*d]
        gate = self.gate(gate_in).softmax(dim=-1)                    # [B, L, 3]

        mixed = (
            gate[..., 0:1] * esm_attended +
            gate[..., 1:2] * prot_attended +
            gate[..., 2:3] * ankh_attended
        )  # [B, L, d_model]
        
        return self.norm(mixed)  

class CrossAttentionMultiModalFusion(nn.Module):
    """
    Cross-attention fusion for three protein modalities:
    - ESM-C sequence embeddings
    - Protein embeddings  
    - Ankh embeddings
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
        
        # Normalization layers
        self.ln_esm = nn.LayerNorm(d_esm)
        self.ln_prot = nn.LayerNorm(d_prot)
        self.ln_ankh = nn.LayerNorm(d_ankh)
        
        # Projection layers
        self.proj_esm = nn.Linear(d_esm, d_out)
        self.proj_prot = nn.Linear(d_prot, d_out)
        self.proj_ankh = nn.Linear(d_ankh, d_out)
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(d_out, dropout, n_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, esm_emb, prot_emb, ankh_emb, pad_mask=None, lengths=None):
        # Normalize and project
        esm_proj = self.proj_esm(self.ln_esm(esm_emb))      # [B, L, d_out]
        prot_proj = self.proj_prot(self.ln_prot(prot_emb))  # [B, L, d_out]
        ankh_proj = self.proj_ankh(self.ln_ankh(ankh_emb))  # [B, L, d_out]
        
        # Apply cross-attention fusion
        fused = self.fusion(esm_proj, prot_proj, ankh_proj, pad_mask)
        
        return self.dropout(fused).masked_fill(pad_mask.unsqueeze(-1), 0.0)

# Drop-in replacement for your current SequenceEncoder
class CrossAttentionSequenceEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 1152,
                 d_prot: int = 128,
                 d_ankh: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        
        self.fusion = CrossAttentionMultiModalFusion(
            d_esm=d_model,
            d_prot=d_prot, 
            d_ankh=d_ankh,
            d_out=d_out,
            dropout=dropout,
            n_heads=n_heads
        )
    
    def forward(self, x, prot_emb, ankh_emb, pad_mask=None, lengths=None):
        return self.fusion(x, prot_emb, ankh_emb, pad_mask)




class SelfAttentionTransformer(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ffn": ResidualFeedForward(d_model, dropout=dropout),
            })
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x_norm = layer["norm"](x)
            y, _ = layer["attn"](x_norm, x_norm, x_norm, key_padding_mask=pad_mask)
            x = x + y
            x = layer["ffn"](x)
        return x



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
        d_msa: int = 768,                     # (unused after MSA removal)
        n_cross_layers: int = 2,              # Transformer self-attention layers
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
        
        # Individual learnable BOS/EOS tokens for each modality
        # ESM-C tokens (d_model dimension)
        self.esm_bos_token = nn.Parameter(torch.zeros(1, d_model))
        self.esm_eos_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.esm_bos_token, std=0.02)
        nn.init.normal_(self.esm_eos_token, std=0.02)
        
        # Protein tokens (d_prot dimension)
        self.prot_bos_token = nn.Parameter(torch.zeros(1, d_prot))
        self.prot_eos_token = nn.Parameter(torch.zeros(1, d_prot))
        nn.init.normal_(self.prot_bos_token, std=0.02)
        nn.init.normal_(self.prot_eos_token, std=0.02)
        
        # Ankh tokens (d_ankh dimension)
        self.ankh_bos_token = nn.Parameter(torch.zeros(1, d_ankh))
        self.ankh_eos_token = nn.Parameter(torch.zeros(1, d_ankh))
        nn.init.normal_(self.ankh_bos_token, std=0.02)
        nn.init.normal_(self.ankh_eos_token, std=0.02)
        
        # Encoders
        self.seq_encoder = CrossAttentionSequenceEncoder(
            d_model=d_model,
            d_prot=d_prot,  # Protein embedding dimension
            d_ankh=d_ankh,  # Ankh3-Large embedding dimension
            d_out=768,
            dropout=dropout,
            n_heads=n_heads
        )
        
        # Replace bidirectional cross-attention (seq↔MSA) with self-attention on fused seq
        self.seq_transformer = SelfAttentionTransformer(
            d_model=768,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout,
        )
        
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

        # Local ESM-C model (trainable by default)
        self.esmc_model = ESMC.from_pretrained("esmc_600m")
        self.esmc_model.train()  # optional, Lightning will set mode each stage anyway

    def setup(self, stage: str) -> None:
        return

    # ---------------------------------------------------------------------
    #  Forward pass - optimized MSA computation
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with on-the-fly ESM-C embeddings and learnable BOS/EOS tokens."""
        # Extract inputs
        sequences: List[str] = batch["sequence"]
        prot_emb = batch["prot_emb"]     # [B, L, d_prot]
        ankh_emb = batch["ankh_emb"]     # [B, L, d_ankh]
        pad_mask = batch["pad_mask"]     # [B, L]
        lengths = batch["lengths"]       # [B]

        # Compute ESM-C embeddings on the fly (includes BOS/EOS replacement inside)
        Lmax = pad_mask.shape[1]
        seq_emb = self._compute_esmc_embeddings(sequences, lengths, Lmax)  # [B, L, d_model]
        # Ensure device/dtype alignment with other modalities
        seq_emb = seq_emb.clone()

        # Insert individual learnable BOS/EOS tokens for protein and ankh modalities
        B = prot_emb.size(0)
        prot_emb[:, 0] = self.prot_bos_token.to(prot_emb.dtype).expand(B, -1)
        ankh_emb[:, 0] = self.ankh_bos_token.to(ankh_emb.dtype).expand(B, -1)
        for i, length in enumerate(lengths):
            eos_pos = int(length.item()) + 1
            if eos_pos < prot_emb.size(1):
                prot_emb[i, eos_pos] = self.prot_eos_token.to(prot_emb.dtype).squeeze()
                ankh_emb[i, eos_pos] = self.ankh_eos_token.to(ankh_emb.dtype).squeeze()

        # Create pooling mask: mask BOS and EOS positions
        pool_mask = pad_mask.clone()
        pool_mask[:, 0] = True
        for i, length in enumerate(lengths):
            eos_pos = int(length.item()) + 1
            if eos_pos < pool_mask.size(1):
                pool_mask[i, eos_pos] = True

        # Encode and transform the fused sequence stream
        seq_z = self.seq_encoder(seq_emb, prot_emb, ankh_emb, pad_mask=pad_mask)
        seq_z = self.seq_transformer(seq_z, pad_mask=pad_mask)

        # Masked mean pooling (excludes BOS/EOS)
        seq_pool = masked_mean_pooling(seq_z, pool_mask)
        logits = self.head(seq_pool)
        return logits, batch["labels"]

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

    def _compute_esmc_embeddings(
        self,
        sequences: List[str],
        lengths: torch.Tensor,
        Lmax: int,
    ) -> torch.Tensor:
        esmc_list: List[torch.Tensor] = []
        for s in sequences:
            prot = ESMProtein(sequence=s)
            h = self.esmc_model.encode(prot)  # [1, L+2, 1152], grad-enabled now
            out = self.esmc_model.logits(
                h, LogitsConfig(sequence=True, return_embeddings=True)
            ).embeddings.squeeze(0)  # [L+2, 1152]
            esmc_list.append(out)

        # Pad each sequence to Lmax along the sequence dimension, then stack
        padded = []
        for e, L in zip(esmc_list, lengths.tolist()):
            copy_len = min(L + 2, e.shape[0], Lmax)
            # truncate if needed, then right-pad to Lmax
            e_trim = e[:copy_len]
            pad_amt = Lmax - e_trim.shape[0]
            if pad_amt > 0:
                e_trim = F.pad(e_trim, (0, 0, 0, pad_amt))  # pad seq dim
            padded.append(e_trim)

        esmc_emb = torch.stack(padded, dim=0)  # [B, Lmax, D] — keeps grad

        # Replace BOS/EOS tokens (OK to do in-place on a grad tensor)
        esmc_emb[:, 0] = self.esm_bos_token.to(esmc_emb.dtype).expand(esmc_emb.size(0), -1)
        for i, L in enumerate(lengths.tolist()):
            eos = L + 1
            if eos < Lmax:
                esmc_emb[i, eos] = self.esm_eos_token.to(esmc_emb.dtype).squeeze(0)

        return esmc_emb