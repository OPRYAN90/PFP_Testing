from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from lightning import LightningModule
from torchmetrics import MeanMetric
from timm.loss import AsymmetricLossMultiLabel  # ASL (multi-label, logit-based)
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
)



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
    """Masked mean pooling over the sequence dimension.

    :param x: [B, L, D] embeddings.
    :param pad_mask: [B, L] boolean mask; True for padding.
    :return: [B, D] pooled embeddings.
    """
    # Zero out padded positions
    x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    
    # Compute lengths (number of non-padded positions)
    lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp_min(1)
    
    # Sum and divide by lengths
    pooled = x_masked.sum(dim=1) / lengths
    
    return pooled
    
class GatedFusion(nn.Module):
    def __init__(self, d_model=768, p_drop=0.1, favor_seq_bias=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model // 2, 2)
        )
    def forward(self, left_feat, right_feat):
        h = torch.cat([left_feat, right_feat], dim=-1)
        logits = self.gate(h)   # [B, 2]
        w = torch.softmax(logits, dim=-1)               # [B, 2]
        return w[..., 0:1] * left_feat + w[..., 1:2] * right_feat

############################################################
#  Encoders for modality pairs
############################################################

class CrossAttentionFusionTwoModalities(nn.Module):
    """Cross-attention between two modalities with softmax gating."""
    
    def __init__(self, d_model: int, attention_dropout: float = 0.1, n_heads: int = 8, fusion_mlp_dropout: float = 0.1):
        super().__init__()
        # Multi-head attention for each modality pair
        self.mod1_to_mod2 = nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
        self.mod2_to_mod1 = nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(fusion_mlp_dropout),
            nn.Linear(d_model // 2, 2)
        ) 
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, mod1, mod2, pad_mask=None):
        # Cross-attention with correct padding masks
        mod1_attended, _ = self.mod1_to_mod2(self.norm1(mod1), self.norm2(mod2), self.norm2(mod2), key_padding_mask=pad_mask)
        mod2_attended, _ = self.mod2_to_mod1(self.norm2(mod2), self.norm1(mod1), self.norm1(mod1), key_padding_mask=pad_mask)
        
        # Convex (softmax) gating per token over the two streams
        gate_in = torch.cat([mod1_attended, mod2_attended], dim=-1)   # [B, L, 2*d]
        gate = self.gate(gate_in).softmax(dim=-1)                    # [B, L, 2]

        mixed = (
            gate[..., 0:1] * mod1_attended +
            gate[..., 1:2] * mod2_attended
        )  # [B, L, d_model]
        
        return mixed

class CrossAttentionTwoStreamFusion(nn.Module):
    """Cross‑attention fusion for two token‑aligned streams."""
    
    def __init__(self,
                 d_in1: int = 1152,
                 d_in2: int = 1024,
                 d_out: int = 768,
                 attention_dropout: float = 0.1,
                 fusion_mlp_dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        # Normalization layers
        self.ln_in1 = nn.LayerNorm(d_in1)
        self.ln_in2 = nn.LayerNorm(d_in2)
        
        # Projection layers
        self.proj_in1 = nn.Linear(d_in1, d_out, bias=False)
        self.proj_in2 = nn.Linear(d_in2, d_out, bias=False)
        
        # Cross-attention fusion (two-way with gating)
        self.fusion = CrossAttentionFusionTwoModalities(d_out, attention_dropout, n_heads, fusion_mlp_dropout)
        
        self.dropout = nn.Dropout(fusion_mlp_dropout)
    
    def forward(self, emb1, emb2, pad_mask=None):
        # Normalize and project
        emb1_proj = self.dropout(self.proj_in1(self.ln_in1(emb1)))  # [B, L, d_out]
        emb2_proj = self.dropout(self.proj_in2(self.ln_in2(emb2)))  # [B, L, d_out]
        
        # Apply cross-attention fusion
        fused = self.fusion(emb1_proj, emb2_proj, pad_mask)
        
        return fused.masked_fill(pad_mask.unsqueeze(-1), 0.0)

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
    
    def forward(self, emb1, emb2, pad_mask=None):
        return self.fusion(emb1, emb2, pad_mask)


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
    """LightningModule for multi‑modal protein function prediction.

    Fuses ESM‑C+Ankh and ProtT5+PGLM token features via cross‑attention and
    sequence‑level AttentionFusion, trained with asymmetric multi‑label loss.
    """
    def __init__(
        self,
        task_type: str = "ec",               # Kept for backward compat; unused in EC metrics
        num_classes: int = 5106,             # <-- required: e.g., len(ec2idx)
        d_esm: int = 1152,                  # Base model dimension
        d_latent: int = 768,                  # Latent dimension for cross-modal fusion and final representation
        d_prot: int = 1024,                   # ProtT5 hidden size (XL UniRef50 = 1024)
        d_ankh: int = 1024,                   # Ankh3-XLarge embedding dimension
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for compatibility and explicit class count
        self.task_type = task_type
        self.num_classes = int(num_classes)
        
        
        
        # Individual learnable BOS/EOS tokens for each modality
        # ESM-C tokens (d_esm dimension) - for pre-computed embeddings
        self.esm_bos_token = nn.Parameter(torch.zeros(1, d_esm))
        self.esm_eos_token = nn.Parameter(torch.zeros(1, d_esm))
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
        
        # Sequence stream: fuse ESM-C with Ankh
        self.seq_encoder = CrossAttentionSequenceEncoder(
            d_in1=d_esm,
            d_in2=d_ankh,
            d_out=d_latent,
            attention_dropout=attention_dropout,
            fusion_mlp_dropout=fusion_mlp_dropout,
            n_heads=n_heads
        )
        # Second stream: ProtT5 + PGLM fusion (generic two-stream)
        self.prot_pglm_encoder = CrossAttentionSequenceEncoder(
            d_in1=d_prot,
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

        # GatedFusion over pooled stream features
        self.fusion = GatedFusion(d_model=d_latent, p_drop=fusion_mlp_dropout)
        
        # Create classifier head for EC
        self.head = MLPHead(d_latent, self.num_classes, head_dropout)

        # For long-tail multi-label, use Asymmetric Loss (ASL)
        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4.0,
            gamma_pos=0.0,   # keep positives strong (paper recommendation)
            clip=0.05,       # ignore very-easy negatives; helps all-zero rows
            eps=1e-8,        # numerical safety (timm default)
        )

        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # Storage for epoch-wise metrics
        self._val_logits = []
        self._val_labels = []
        self._test_logits = []
        self._test_labels = []
        self._test_pids: List[str] = []
        self._test_threshold: float = 0.5  # will be set from a one-time sweep on val before testing

    def setup(self, stage: str) -> None:
        """No-op setup."""
        return

    def _insert_bos_eos_tokens(
        self,
        seq_emb: torch.Tensor,
        ankh_emb: torch.Tensor,
        prot_emb: torch.Tensor,
        pglm_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """Insert per‑modality BOS at position 0 and EOS at position L+1 (in‑place)."""
        batch_size = seq_emb.size(0)

        # BOS at position 0
        seq_emb[:, 0] = self.esm_bos_token.to(seq_emb.dtype).expand(batch_size, -1)
        ankh_emb[:, 0] = self.ankh_bos_token.to(ankh_emb.dtype).expand(batch_size, -1)
        prot_emb[:, 0] = self.prot_bos_token.to(prot_emb.dtype).expand(batch_size, -1)
        pglm_emb[:, 0] = self.pglm_bos_token.to(pglm_emb.dtype).expand(batch_size, -1)

        # EOS at position L+1
        for i, length in enumerate(lengths):
            eos_pos = length + 1
            if eos_pos < seq_emb.size(1):
                seq_emb[i, eos_pos] = self.esm_eos_token.to(seq_emb.dtype).squeeze()
                ankh_emb[i, eos_pos] = self.ankh_eos_token.to(ankh_emb.dtype).squeeze()
                prot_emb[i, eos_pos] = self.prot_eos_token.to(prot_emb.dtype).squeeze()
                pglm_emb[i, eos_pos] = self.pglm_eos_token.to(pglm_emb.dtype).squeeze()

    def _build_pool_mask(self, pad_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return pooling mask with BOS/EOS positions masked out (True)."""
        pool_mask = pad_mask.clone()
        pool_mask[:, 0] = True
        for i, length in enumerate(lengths):
            eos_pos = length + 1
            if eos_pos < pool_mask.size(1):
                pool_mask[i, eos_pos] = True
        return pool_mask

    # ---------------------------------------------------------------------
    #  Forward pass
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with per‑modality learnable BOS/EOS tokens and dual‑stream fusion (L-->L+2).

        :param batch: Input dict containing embeddings, masks, lengths, and labels.
        :return: Logits and ground‑truth labels.
        """
        # Extract inputs - all embeddings are pre-computed
        seq_emb = batch["esmc_emb"]      # ESM-C: [B, L, d_esm]
        ankh_emb = batch["ankh_emb"]     # Ankh3-XLarge: [B, L, d_ankh]
        prot_emb = batch["prot_emb"]     # ProtT5: [B, L, d_prot]
        pglm_emb = batch["pglm_emb"]     # PGLM: [B, L, d_pglm]
        pad_mask = batch["pad_mask"]     # [B, L] True where padded
        lengths = batch["lengths"]       # [B]

        # Preprocess embeddings and build pooling mask
        self._insert_bos_eos_tokens(seq_emb, ankh_emb, prot_emb, pglm_emb, lengths)
        pool_mask = self._build_pool_mask(pad_mask, lengths)
        
        # Sequence stream: ESM‑C + Ankh
        seq_z = self.seq_encoder(seq_emb, ankh_emb, pad_mask=pad_mask)  # [B, L, d]

        # ProtT5 + PGLM stream
        prot_pglm_z = self.prot_pglm_encoder(prot_emb, pglm_emb, pad_mask=pad_mask)  # [B, L, d]

        # Cross‑modal attention between streams
        seq_z, prot_pglm_z = self.cross_attn(seq_z, prot_pglm_z, pad_mask, pad_mask)

        # Masked mean pooling (exclude BOS/EOS)
        seq_pool = masked_mean_pooling(seq_z, pool_mask)
        aux_pool = masked_mean_pooling(prot_pglm_z, pool_mask)

        # GatedFusion between streams
        fused = self.fusion(seq_pool, aux_pool)

        logits = self.head(fused)                  # [B, C]
        return logits, batch["labels"]

    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        self.train_loss(loss)
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
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.val_loss(loss)
        # Store for epoch‑wise CAFA metrics computation (convert to fp32 for numpy)
        self._val_logits.append(logits.detach().float().cpu())   # keep raw, no sigmoid
        self._val_labels.append(labels.detach().float().cpu())
        
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.test_loss(loss)
 
        # Store for epoch‑wise CAFA metrics computation (convert to fp32 for numpy)
        self._test_logits.append(logits.detach().float().cpu())   # keep raw, no sigmoid
        self._test_labels.append(labels.detach().float().cpu())
        # Track protein IDs to align with stacked logits order
        if "protein_id" in batch:
            self._test_pids.extend(list(batch["protein_id"]))

    def _compute_ec_metrics(self, logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute EC comparison metrics:
          - AUC (samples-averaged ROC-AUC)
          - Precision/Recall/F1 at a fixed threshold (samples-averaged)
        Robust to degenerate batches with all-zero labels.
        """
        y_true = labels.detach().cpu().numpy().astype(int)
        y_score = torch.sigmoid(logits).detach().cpu().numpy()
        # Thresholded preds for PR/F1 (protein-centric)
        y_pred = (y_score >= threshold).astype(int)

        # Some folds can be degenerate => guard AUC
        auc_samples = float(roc_auc_score(y_true, y_score, average="samples"))
        prec = float(precision_score(y_true, y_pred, average="samples", zero_division=0))
        rec  = float(recall_score( y_true, y_pred, average="samples", zero_division=0))
        f1   = float(f1_score(      y_true, y_pred, average="samples", zero_division=0))
        return {"AUC": auc_samples, "Precision": prec, "Recall": rec, "F1": f1}

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits) 
        labels = torch.cat(self._val_labels) 
        out = {"val/loss": float(self.val_loss.compute())}
        # Validation monitors ONLY micro-AUPR (no thresholding)
        y_true  = labels.detach().cpu().numpy().astype(int)
        y_score = torch.sigmoid(logits).detach().cpu().numpy()
        aupr_micro = float(average_precision_score(y_true, y_score, average="micro"))
        out["val/AUPR_micro"] = aupr_micro
        self.log_dict(out, prog_bar=True, sync_dist=True)

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

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        """
        Before running the test loop, sweep the **validation set once**
        to find a single global threshold that maximizes F1 (samples-averaged).
        """
        dm = self.trainer.datamodule
        if dm is None:
            return
        self.eval()
        val_loader = dm.val_dataloader()
        all_logits, all_labels = [], []
        print(f"Val loader length: {len(val_loader)}")
        print("Validating for threshold...")
        for batch in val_loader:
            # move tensors to the same device as module
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)
            logits, labels = self.forward(batch)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels)

        # Sweep thresholds and pick the argmax F1 (samples-averaged)
        print("Finished validating for threshold. Now sweeping thresholds...")
        y_true  = val_labels.numpy().astype(int)
        y_score = torch.sigmoid(val_logits).numpy()
        print(f"AUPR Micro: {float(average_precision_score(y_true, y_score, average='micro'))}")
        best_t, best_f1 = 0.5, -1.0
        for t in [i/100 for i in range(1, 100)]:  # 0.01 ... 0.99
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._test_threshold = float(best_t)
        # Optional: expose for logs
        self.log("test/threshold", self._test_threshold, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        logits = torch.cat(self._test_logits) 
        labels = torch.cat(self._test_labels) 
        out = {"test/loss": float(self.test_loss.compute())}
        out |= {f"test/{k}": v for k, v in self._compute_ec_metrics(logits, labels, threshold=self._test_threshold).items()}
        self.log_dict(out, prog_bar=True, sync_dist=True)

        # Save lightweight outputs: probabilities, EC terms, and PIDs
        try:
            if getattr(self.trainer, "global_rank", 0) == 0:
                probs = torch.sigmoid(logits).cpu()
                # best-effort: fetch EC label order from datamodule mapping
                ec_terms = []
                if hasattr(self.trainer.datamodule, "_ec2idx"):
                    # invert mapping to index order
                    inv = {v: k for k, v in self.trainer.datamodule._ec2idx.items()}
                    ec_terms = [inv[i] for i in range(len(inv))]
                logs_dir = Path("/teamspace/studios/this_studio/PFP_Testing/logs")
                logs_dir.mkdir(parents=True, exist_ok=True)
                out_path = logs_dir / "ec_preds.pt"
                torch.save({
                    "task_type": "ec",
                    "pids": list(self._test_pids),
                    "ec_terms": ec_terms,
                    "probs": probs,
                    "labels": labels,
                    "threshold": self._test_threshold,
                }, out_path)
        except Exception as e:
            # Non-fatal: continue even if saving fails
            print(f"[Warn] Failed to save test outputs: {e}")

        self.test_loss.reset(); self._test_logits.clear(); self._test_labels.clear(); self._test_pids.clear()