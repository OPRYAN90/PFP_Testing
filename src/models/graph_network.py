from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torch_geometric.utils import to_undirected

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


class MLPHead(nn.Module):
    """Final classification head with layer norm and dropout."""
    
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1): 
        super().__init__()
        # assert d_in == 768, "d_in must be 768"
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 602),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(602, d_out), #EXTREME WARNING NOTE: DIMS DEPEDNING ON TASK AND D_MSA
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------ Simple edge dropout utility -----------------
def drop_edge(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0: 
        return edge_index
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p
    return edge_index[:, keep]

# ------------ One GCN block (conv + norm + relu + residual) ------------
class GCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, cached=False, normalize=True)
        self.norm = GraphNorm(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        out = self.conv(x, edge_index)
        out = self.norm(out, batch)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # dims are fixed to d_in, so residual always valid
        out = out + x
        return out

# ------------ Full network -----------------
class GCNNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_layers: int,
        dropout: float = 0.2,
        edge_drop: float = 0.0,
    ):
        super().__init__()
        self.edge_drop = edge_drop

        dims = [d_in] * num_layers  # keep feature dim constant
        self.layers = nn.ModuleList([
            GCNBlock(dims[i], dims[i + 1], dropout) for i in range(num_layers - 1)
        ])

    def forward(self, x, edge_index, batch):
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        edge_index = drop_edge(edge_index, self.edge_drop)

        for layer in self.layers:
            x = layer(x, edge_index, batch)

        g = global_mean_pool(x, batch)      # [B, d_in]
        return g

############################################################
#  LightningModule - Updated with Canonical Evoformer-Lite
############################################################

class ProteinLitModule(LightningModule):
    def __init__(
        self,
        task_type: str,
        d_in: int = 1152,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_drop: float = 0.0,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        warmup_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for easy access
        self.task_type = task_type

        # GCN backbone -> pooled graph embedding
        self.model = GCNNet(
            d_in=d_in,
            num_layers=num_layers,
            dropout=dropout,
            edge_drop=edge_drop,
        )

        # Final MLP head on graph embedding
        self.head = MLPHead(
            d_in=d_in,
            d_out=get_num_classes_for_task(task_type),
            dropout=dropout,
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

    # no MSA model anymore, so no special setup needed
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    # ---------------------------------------------------------------------
    #  Forward pass with EOS token scatter
    # ---------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """GCN forward: pooled graph embedding -> MLP head -> logits."""
        g = self.model(batch["x"], batch["edge_index"], batch["graph_batch"])
        logits = self.head(g)
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


