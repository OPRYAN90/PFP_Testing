import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import AUROC, AveragePrecision, MaxMetric, MeanMetric

# NOTE: We avoid hard dependencies on external graph libraries (e.g. PyG) so the
# template keeps working out-of-the-box. If you install torch-geometric you can
# easily swap the naive `GraphConv` below with `GCNConv`/`GATConv`.

############################################################
#  Low-level building blocks
############################################################

class RowDropout(nn.Module):
    """Zero out individual MSA rows (sequences) without rescaling.
    
    Note: The first row (query sequence) is never dropped out.
    """
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Sample mask of shape [B, N_seq, 1, 1] to drop individual sequences
        mask = torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.p
        # Ensure first row (query sequence) is never dropped
        mask[:, 0, :, :] = 1.0  # Always keep the query sequence
        return x * mask  # no 1/(1-p) rescaling for row dropout


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
    
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1): #TODO: CHECK DIMS
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, d_out),
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
        d_model: int = 640,  # ESM-C embedding dimension
        n_layers: int = 2,   # Additional transformer layers
        n_heads: int = 8,    # Attention heads
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Additional transformer layers on top of ESM-C
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with pre-computed ESM-C embeddings.
        
        Args:
            x: Pre-computed ESM-C embeddings [B, L, d_model]
            src_key_padding_mask: [B, L] True where padded
            
        Returns:
            Enhanced sequence representations [B, L, d_model]
        """
        # Apply additional transformer layers with padding mask
        x = self.tr(x, src_key_padding_mask=src_key_padding_mask)
        
        return x  # [B, L, d_model]


class MSAEncoder(nn.Module):
    """Encoder for Multiple Sequence Alignment (MSA) embeddings.

    Expected input: msa [B, N_seq, L_cls+res, d_msa] (already includes CLS at col 0)
    - B: batch size
    - N_seq: alignment depth (≈256) 
    - L_cls+res: CLS token + residue length
    - d_msa: embedding dim from MSA-Transformer (≈768)

    Processing steps:
    0. Append learnable EOS token → [B, N_seq, L_cls+res+eos, d_msa]
    1. Row-dropout – randomly drop individual homologous sequences
    2. Channel dropout – AlphaFold-style embedding channel dropout
    3. Learn conservation scores over the alignment depth  
    4. Softmax-weighted average → [B, L_cls+res+eos, d_msa]
    5. Linear projection → d_model, feature-dropout
    """
    
    def __init__(
        self,
        d_msa: int = 768,
        d_model: int = 640,
        p_row: float = 0.15,    # dropout prob for individual MSA rows
        p_chan: float = 0.15,   # channel dropout prob 
        p_feat: float = 0.10,   # feature dropout after projection
    ):
        super().__init__()
        
        # 0) Learnable EOS token shared across rows
        self.eos_token = nn.Parameter(torch.zeros(1, 1, 1, d_msa))
        nn.init.normal_(self.eos_token, std=0.02)
        
        # 1) Row-level dropout (zeros individual sequences)
        self.row_dropout = RowDropout(p_row)
        
        # 2) Channel dropout (zeros embedding channels, mask shared across rows)
        self.channel_dropout = RowwiseChannelDropout(p_chan)
        
        # 3) Conservation scoring head
        self.conservation_head = nn.Linear(d_msa, 1, bias=False)
        
        # 4) Projection + regularization
        self.proj = nn.Linear(d_msa, d_model, bias=False)
        self.feat_dropout = nn.Dropout(p_feat)
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
        eos_idx = lengths + 1                      # [B]

        B, N_seq, _, D = msa.shape
        # Build broadcastable index tensors
        batch_idx = torch.arange(B,  device=msa.device)[:, None, None]
        seq_idx   = torch.arange(N_seq, device=msa.device)[None, :, None]
        # eos_idx already has shape [B]; add singleton dims
        pos_idx   = eos_idx[:, None, None]
        #TODO Consider changing graph process 
        # FIX: Handle in-place writes with torch.compile (graph break approach)
        if torch.compiler.is_compiling():  # cheaper than .clone()
            torch._dynamo.graph_break()   # forces eager mode for next operation
        msa[batch_idx, seq_idx, pos_idx, :] = self.eos_token

        # ------------------------------------------------------------
        # 1. Row dropout  ─── 2. Channel dropout  ─── 3-5. Rest
        # ------------------------------------------------------------
        msa     = self.row_dropout(msa)
        msa     = self.channel_dropout(msa)

        scores  = self.conservation_head(msa).squeeze(-1)     # [B, N_seq, L_pad]
        
        # Mask out padded positions before softmax
        # pad_mask: [B, L] → expand to [B, N_seq, L]
        mask = pad_mask.unsqueeze(1).expand(-1, msa.size(1), -1)      
        scores   = scores.masked_fill(mask, -1e4)                  # never get prob-mass
        drop_mask = (msa.abs().sum(-1) == 0)                           # [B,N_seq,L]
        mask_logits = drop_mask & (~mask)                          # only real residues
        scores = scores.masked_fill(mask_logits, -1e4)

        weights = torch.softmax(scores, dim=1)                # over N_seq
        weights = weights.masked_fill(mask, 0.0)       # ← explicit zero-out after softmax
        pooled  = (msa * weights.unsqueeze(-1)).sum(dim=1)    # [B, L_pad, d_msa]
        x = self.norm(pooled)
        x = self.proj(x)
        x = self.feat_dropout(x)
        return x #TODO: FFN?


############################################################
#  Cross-modal attention fusion
############################################################

class CrossModalAttention(nn.Module):
    """Attend each modality to the other modality.

    This is a simple stacked MultiHeadAttention mechanism for ESM and MSA modalities only.
    Uses Pre-LN for better gradient stability and bi-directional cross-attention.
    """

    def __init__(self, d_model: int = 640, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    "seq_norm": nn.LayerNorm(d_model),
                    "msa_norm": nn.LayerNorm(d_model),
                    "seq_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    "msa_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    "seq_ffn": ResidualFeedForward(d_model, dropout=dropout),
                    "msa_ffn": ResidualFeedForward(d_model, dropout=dropout),
                })
            )

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
#  LightningModule - Adapted for Lightning-Hydra Setup
############################################################

class ProteinLitModule(LightningModule):
    """LightningModule for multi-modal protein function prediction.

    Adapted for Lightning-Hydra template with dynamic num_classes and 
    Hydra-compatible optimizer/scheduler configuration.
    
    TODO: Research Director - This integrates with your datamodule's num_classes property
    and handles your specific batch format with CB distance matrices.
    """

    def __init__(
        self,
        d_model: int = 640,           # Base model dimension
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
        scheduler: torch.optim.lr_scheduler = None,  # Hydra scheduler config  
        debugging: bool = True,       # Default to debugging mode (disables compilation)
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
        
        # DON'T create head here - wait for setup()
        self.head = None  # Will be created in setup()

        # Note: BCEWithLogitsLoss doesn't support label_smoothing (only CrossEntropyLoss does)
        # For multi-label protein prediction, we use standard BCE loss
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics - will be updated in setup() with correct num_classes
        # Use placeholder values, will be recreated in setup()
        self.train_auroc = None
        self.val_auroc = None
        self.test_auroc = None  
        self.val_ap = None
        
        # Loss tracking (these don't depend on num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_auroc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """Setup hook to get num_classes from datamodule and update components."""
        if hasattr(self.trainer.datamodule, 'num_classes'):
            num_classes = self.trainer.datamodule.num_classes
            
            # Create classifier head (ONLY TIME IT'S CREATED)
            self.head = MLPHead(self.hparams.d_model, num_classes, self.hparams.dropout)
            
            # Create metrics with correct number of classes
            self.train_auroc = AUROC(task="multilabel", num_labels=num_classes, average="macro")
            self.val_auroc = AUROC(task="multilabel", num_labels=num_classes, average="macro")
            self.test_auroc = AUROC(task="multilabel", num_labels=num_classes, average="macro")
            self.val_ap = AveragePrecision(task="multilabel", num_labels=num_classes, average="macro")
            
            print(f"Model configured for {num_classes} classes")
        else:
            raise ValueError("Datamodule must have 'num_classes' attribute")
        
        # Compilation logic: Compile when NOT debugging TODO: WARNING DOUBLE DIP
        if stage == "fit":
            if not self.hparams.debugging:
                print("🚀  Production mode: compiling heavy encoders")
                self.seq_encoder = torch.compile(self.seq_encoder)
                self.msa_encoder = torch.compile(self.msa_encoder)
                self.cross_attn = torch.compile(self.cross_attn)
            else:
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
        #TODO: UPDATE TO USE MSA LENGTHS   
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
        preds = torch.sigmoid(logits)
        self.train_auroc(preds, labels.int())
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.val_loss(loss)
        preds = torch.sigmoid(logits)
        self.val_auroc(preds, labels.int())
        self.val_ap(preds, labels.int())
        
        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        
        # Update metrics
        self.test_loss(loss)
        preds = torch.sigmoid(logits)
        self.test_auroc(preds, labels.int())
        
        # Log metrics  
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        ap = self.val_ap.compute()
        self.val_auroc_best(auroc)
        
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/ap", ap, prog_bar=False, sync_dist=True)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset validation metrics to avoid storing results from sanity checks
        self.val_loss.reset()
        self.val_auroc.reset() 
        self.val_ap.reset()
        self.val_auroc_best.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers - adapted for Hydra config system."""
        # TODO: Research Director - This integrates with your Hydra optimizer configs
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass  # Future: add epoch-end logic here

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass  # Future: add test cleanup logic here


############################################################
#  Developer notes / open questions for Research Director
############################################################
#  TODO: Research Director - Key integration points that need your input:
#  
#  1. ESM-C Integration: Currently using MSA embeddings as placeholder for sequence.
#     Need to implement actual ESM-C embedding in SequenceEncoder.forward()
#  
#  2. MSA Shape: What's the exact shape of your MSA embeddings? 
#     - [B, L, d_msa] (already processed) or [B, N_seq, L, d_msa] (raw)?
#  
#  3. Distance Matrix Priority: CB vs CA distances?
#     - CB provides better side-chain interaction info
#     - CA is more standard/available
#     - Currently defaulting to CB with CA fallback
#  
#  4. Batch Handling: Your datamodule uses batch_size=1 initially.
#     - Single-sample processing works fine
#     - Batch processing needs collate_fn implementation
#  
#  5. Class Imbalance: Multi-label protein function prediction often has severe imbalance.
#     - Consider per-class weights in BCEWithLogitsLoss
#     - Focal loss might be beneficial
#  
#  6. Model Dimensions: Verify d_model=640 matches your embeddings
#     - ESM-C models vary: ESM-2 650M uses 1280 dim
#     - MSA-Transformer uses 768 dim
#     - Current projection layers handle dimension mismatches


if __name__ == "__main__":
    # Test model instantiation
    model = ProteinLitModule()
    print("ProteinLitModule created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")