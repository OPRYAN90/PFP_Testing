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
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
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
    """Cross-attention between four modalities with softmax gating (esm, prot, ankh, msa)"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention for each modality attending to the other three
        self.esm_to_others  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prot_to_others = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ankh_to_others = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.msa_to_others  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Per-token gating over 4 attended streams
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4)
        )
        
        # Final combination layer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, esm, prot, ankh, msa, pad_mask=None):
        # Build concatenated key/value banks for each query stream
        others_esm  = torch.cat([prot, ankh, msa], dim=1)  # [B, 3L, d]
        others_prot = torch.cat([esm, ankh, msa], dim=1)  # [B, 3L, d]
        others_ankh = torch.cat([esm, prot, msa], dim=1)  # [B, 3L, d]
        others_msa  = torch.cat([esm, prot, ankh], dim=1) # [B, 3L, d]
        
        if pad_mask is not None:
            triple = [pad_mask, pad_mask, pad_mask]
            others_esm_mask  = torch.cat(triple, dim=1)
            others_prot_mask = torch.cat(triple, dim=1)
            others_ankh_mask = torch.cat(triple, dim=1)
            others_msa_mask  = torch.cat(triple, dim=1)
            # Zero out padded positions in query tensors
            esm  = esm.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            prot = prot.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            ankh = ankh.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            msa  = msa.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        else:
            others_esm_mask = others_prot_mask = others_ankh_mask = others_msa_mask = None
        
        # Cross-attention per stream
        esm_attended,  _ = self.esm_to_others(esm,  others_esm,  others_esm,  key_padding_mask=others_esm_mask)
        prot_attended, _ = self.prot_to_others(prot, others_prot, others_prot, key_padding_mask=others_prot_mask)
        ankh_attended, _ = self.ankh_to_others(ankh, others_ankh, others_ankh, key_padding_mask=others_ankh_mask)
        msa_attended,  _ = self.msa_to_others(msa,  others_msa,  others_msa,  key_padding_mask=others_msa_mask)
        
        # Softmax gating across four streams
        gate_in = torch.cat([esm_attended, prot_attended, ankh_attended, msa_attended], dim=-1)   # [B, L, 4*d]
        gate = self.gate(gate_in).softmax(dim=-1)                                                 # [B, L, 4]

        mixed = (
            gate[..., 0:1] * esm_attended +
            gate[..., 1:2] * prot_attended +
            gate[..., 2:3] * ankh_attended +
            gate[..., 3:4] * msa_attended
        )  # [B, L, d_model]
        
        return self.norm(mixed)

class CrossAttentionMultiModalFusion(nn.Module):
    """
    Cross-attention fusion for four protein modalities:
    - ESM-C sequence embeddings
    - Protein embeddings  
    - Ankh embeddings
    - MSA encoder outputs (LN + proj to d_out)
    """
    
    def __init__(self,
                 d_esm: int = 1152,
                 d_prot: int = 1024,
                 d_ankh: int = 1024,
                 d_msa: int = 768,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        self.d_out = d_out
        
        # Normalization layers
        self.ln_esm = nn.LayerNorm(d_esm)
        self.ln_prot = nn.LayerNorm(d_prot)
        self.ln_ankh = nn.LayerNorm(d_ankh)
        self.ln_msa = nn.LayerNorm(d_msa)
        
        # Projection layers
        self.proj_esm = nn.Linear(d_esm, d_out)
        self.proj_prot = nn.Linear(d_prot, d_out)
        self.proj_ankh = nn.Linear(d_ankh, d_out)
        self.proj_msa = nn.Linear(d_msa, d_out)
        
        # Cross-attention fusion (now 4-way)
        self.fusion = CrossAttentionFusion(d_out, dropout, n_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, esm_emb, prot_emb, ankh_emb, msa_emb, pad_mask=None, lengths=None):
        # Normalize and project
        esm_proj  = self.proj_esm(self.ln_esm(esm_emb))       # [B, L, d_out]
        prot_proj = self.proj_prot(self.ln_prot(prot_emb))    # [B, L, d_out]
        ankh_proj = self.proj_ankh(self.ln_ankh(ankh_emb))    # [B, L, d_out]
        msa_proj  = self.proj_msa(self.ln_msa(msa_emb))       # [B, L, d_out]
        
        # Apply cross-attention fusion
        fused = self.fusion(esm_proj, prot_proj, ankh_proj, msa_proj, pad_mask)
        
        return self.dropout(fused).masked_fill(pad_mask.unsqueeze(-1), 0.0)

# Drop-in replacement for your current SequenceEncoder
class CrossAttentionSequenceEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 1152,
                 d_prot: int = 1024,
                 d_ankh: int = 1024,
                 d_out: int = 768,
                 dropout: float = 0.1,
                 n_heads: int = 8):
        super().__init__()
        
        self.fusion = CrossAttentionMultiModalFusion(
            d_esm=d_model,
            d_prot=d_prot, 
            d_ankh=d_ankh,
            d_msa=768,
            d_out=d_out,
            dropout=dropout,
            n_heads=n_heads
        )
    
    def forward(self, x, prot_emb, ankh_emb, msa_emb, pad_mask=None, lengths=None):
        return self.fusion(x, prot_emb, ankh_emb, msa_emb, pad_mask)





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

class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "drop": nn.Dropout(dropout),
                "ffn": ResidualFeedForward(d_model, dropout=dropout),
            })
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer["norm"](x)
            attn_out, _ = layer["attn"](x, x, x, key_padding_mask=pad_mask)
            x = x + layer["drop"](attn_out)
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
        d_prot: int = 1024,                   # ProtT5 embedding dimension
        d_ankh: int = 1024,                   # Ankh3-Large embedding dimension
        d_msa: int = 768,                     # MSA embedding dimension
        n_cross_layers: int = 2,              # Cross-attention layers
        n_heads: int = 8,                     # Attention heads
        dropout: float = 0.1,                 # Dropout rate
        optimizer: Optional[Any] = None,      # Hydra optimizer config
        scheduler: Optional[Any] = None,      # Hydra scheduler config  
        warmup_ratio: float = 0.05,           # Warmup ratio for cosine schedule (5% of total training steps)
        # --- LoRA toggles (placeholders; wiring to be added later) ---
        lora_esmc: bool = False,
        lora_prott5: bool = False,
        lora_ankh: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        # Store task type for easy access
        self.task_type = task_type
        
        # Load models (frozen by default) ---------------------------------
        import esm
        self.msa_model, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_model.eval().requires_grad_(False)

        # ESM-C 600M (SDK)
        self.esmc_model = ESMC.from_pretrained("esmc_600m").eval().requires_grad_(False)

        # ProtT5 (HF)
        self.prot_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval().requires_grad_(False)

        # Ankh3-Large (HF)
        self.ankh_enc = T5EncoderModel.from_pretrained("ElnaggarLab/ankh3-large").eval().requires_grad_(False)

        # --- LoRA placeholders: will attach adapters here later ---
        if self.hparams.lora_esmc:
            self._attach_lora_placeholder(self.esmc_model, "esmc")
        if self.hparams.lora_prott5:
            self._attach_lora_placeholder(self.prot_t5, "prot_t5")
        if self.hparams.lora_ankh:
            self._attach_lora_placeholder(self.ankh_enc, "ankh")
        
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
        
        # MSA tokens (d_msa dimension) - for insertion after MSA encoder
        self.msa_bos_token = nn.Parameter(torch.zeros(1, d_msa))
        self.msa_eos_token = nn.Parameter(torch.zeros(1, d_msa))
        nn.init.normal_(self.msa_bos_token, std=0.02)
        nn.init.normal_(self.msa_eos_token, std=0.02)
        
        # Encoders
        self.seq_encoder = CrossAttentionSequenceEncoder(
            d_model=d_model,
            d_prot=d_prot,  # ProtT5 embedding dimension
            d_ankh=d_ankh,  # Ankh3-Large embedding dimension
            d_out=768,
            dropout=dropout,
            n_heads=n_heads
        )
        self.msa_encoder = MSAEncoder(
            d_msa=d_msa,
        )

        # Self-attention transformer over fused representation (replaces bidirectional cross-attention)
        self.self_attn = SelfAttentionEncoder(
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
        """Forward with on-the-fly PLM embeddings (ESM-C, ProtT5, Ankh).
        Datamodule passes tokens; we keep BOS/EOS through the PLMs and
        only replace them with learnable tokens afterward."""
        sequences = batch["sequence"]              # List[str], length B
        prot_ids = batch["prot_input_ids"]         # [B, T_max]
        prot_mask = batch["prot_attention_mask"]   # [B, T_max]
        ankh_ids = batch["ankh_input_ids"]         # [B, T_max]
        ankh_mask = batch["ankh_attention_mask"]   # [B, T_max]
        msa_tok_list = batch["msa_tok"]            # list[Tensor] – variable [N_i, L_i]

        pad_mask = batch["pad_mask"]               # [B, L_max] (True where padded)
        seq_pad    = batch["seq_pad_mask"]  # [B, N_max]

        # Shapes / sizes
        B = len(sequences)
        lengths = batch["lengths"]                 # [B] (residue counts, no specials)
        Lmax = pad_mask.shape[1]                   # = max(lengths) + 2

        # -------------------------------------------------------------
        # 1) Compute MSA embeddings (unchanged)
        # -------------------------------------------------------------
        target_len = Lmax
        msa_emb, msa_compute_time = self._compute_msa_embeddings(msa_tok_list, target_len)  # [B, N_seq_max, target_len, d_msa]
        # Expose timing so callbacks (e.g., BatchTimer) can log it **after** the forward pass.
        # Storing it on `self` ensures it is available in hooks such as `on_before_backward`.
        self._last_msa_compute_time = msa_compute_time
        # Keep original behaviour for potential external use
        batch["msa_compute_time"] = msa_compute_time
        # For conservation head: treat BOS as masked in MSA encoder
        msa_emb = msa_emb.clone()
        msa_emb[:, :, 0] = 0.0

        # Create mask for MSA encoder: mask BOS position (0) and ensure EOS positions are masked
        msa_encoder_mask = pad_mask.clone()
        msa_encoder_mask[:, 0] = True  # Mask out BOS position for MSA encoder (padding=True)
        
        # Also ensure EOS positions (L+1) are masked as True
        for i, length in enumerate(lengths):
            eos_pos = length + 1
            if eos_pos < msa_encoder_mask.size(1):  # Safety check
                msa_encoder_mask[i, eos_pos] = True  # Mask out EOS position for MSA encoder

        # -------------------------------------------------------------
        # 2) PLM embeddings on-the-fly via helpers (with BOS/EOS injection)
        # -------------------------------------------------------------
        esmc_emb = self._compute_esmc_embeddings(sequences, lengths, Lmax)
        prot_emb = self._compute_prott5_embeddings(prot_ids, prot_mask, lengths, Lmax)
        ankh_emb = self._compute_ankh_embeddings(ankh_ids, ankh_mask, lengths, Lmax)

        # -------------------------------------------------------------
        # 3) MSA encode (unchanged)
        # -------------------------------------------------------------
        msa_z = self.msa_encoder(msa_emb, pad_mask=msa_encoder_mask, seq_pad=seq_pad)  # [B, L, d_msa]

        # Insert BOS/EOS tokens for MSA stream
        msa_z[:, 0]    = self.msa_bos_token.to(msa_z.dtype).expand(B, -1)
        for i, L in enumerate(lengths.tolist()):
            eos = L + 1
            if eos < Lmax:
                msa_z[i, eos]    = self.msa_eos_token.to(msa_z.dtype).squeeze()

        # Multi-modal fusion over four modalities → [B, L, 768]
        fused_z = self.seq_encoder(esmc_emb, prot_emb, ankh_emb, msa_z, pad_mask=pad_mask)

        # Self-attention transformer over fused tokens
        fused_z = self.self_attn(fused_z, pad_mask=pad_mask)

        # Masked mean pooling (exclude BOS/EOS via msa_encoder_mask)
        pooled = masked_mean_pooling(fused_z, msa_encoder_mask)  # [B, 768]

        logits = self.head(pooled)                  # [B, C]
        
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
    #  Embedding helpers (inference mode) with BOS/EOS injection
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _compute_esmc_embeddings(
        self,
        sequences: List[str],
        lengths: torch.Tensor,
        Lmax: int,
    ) -> torch.Tensor:
        esmc_list: List[torch.Tensor] = []
        for s in sequences:
            prot = ESMProtein(sequence=s)
            h = self.esmc_model.encode(prot)  # [1, L+2, 1152]
            out = self.esmc_model.logits(h, LogitsConfig(sequence=True, return_embeddings=True)).embeddings  # [1, L+2, 1152]
            esmc_list.append(out.squeeze(0))  # [L+2, 1152]
        B = len(sequences)
        D = esmc_list[0].shape[-1]
        esmc_emb = esmc_list[0].new_zeros(B, Lmax, D)
        for i, (e, L) in enumerate(zip(esmc_list, lengths.tolist())):
            copy_len = min(L + 2, e.shape[0], Lmax)
            esmc_emb[i, :copy_len] = e[:copy_len]
            # Replace BOS/EOS with learnable tokens
            esmc_emb[i, 0] = self.esm_bos_token.to(esmc_emb.dtype).squeeze(0)
            eos = L + 1
            if eos < Lmax:
                esmc_emb[i, eos] = self.esm_eos_token.to(esmc_emb.dtype).squeeze(0)
        return esmc_emb

    @torch.inference_mode()
    def _compute_prott5_embeddings(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,   # [B, T]
        lengths: torch.Tensor,
        Lmax: int,
    ) -> torch.Tensor:
        proth = self.prot_t5(input_ids=input_ids.to(self.prot_t5.device),
                              attention_mask=attention_mask.to(self.prot_t5.device)).last_hidden_state  # [B, T, 1024]
        B, _, D = proth.shape
        prot_emb = proth.new_zeros(B, Lmax, D)
        for i, L in enumerate(lengths.tolist()):
            # Fill residues into positions 1..L
            prot_emb[i, 1:L+1] = proth[i, :L, :]
            # Inject BOS/EOS learnable tokens
            prot_emb[i, 0] = self.prot_bos_token.to(prot_emb.dtype).squeeze(0)
            eos = L + 1
            if eos < Lmax:
                prot_emb[i, eos] = self.prot_eos_token.to(prot_emb.dtype).squeeze(0)
        return prot_emb

    @torch.inference_mode()
    def _compute_ankh_embeddings(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,   # [B, T]
        lengths: torch.Tensor,
        Lmax: int,
    ) -> torch.Tensor:
        ankhh = self.ankh_enc(input_ids=input_ids.to(self.ankh_enc.device),
                              attention_mask=attention_mask.to(self.ankh_enc.device)).last_hidden_state  # [B, T, 1024]
        B, _, D = ankhh.shape
        ankh_emb = ankhh.new_zeros(B, Lmax, D)
        for i, L in enumerate(lengths.tolist()):
            # Model outputs: first token is prefix, last is EOS; map residues into 1..L
            ankh_emb[i, 1:L+1] = ankhh[i, 1:L+1, :]
            # Inject BOS/EOS learnable tokens
            ankh_emb[i, 0] = self.ankh_bos_token.to(ankh_emb.dtype).squeeze(0)
            eos = L + 1
            if eos < Lmax:
                ankh_emb[i, eos] = self.ankh_eos_token.to(ankh_emb.dtype).squeeze(0)
        return ankh_emb

    # -------------------------------------------------------------
    #  LoRA hook (placeholder)
    # -------------------------------------------------------------
    def _attach_lora_placeholder(self, model: nn.Module, tag: str) -> None:
        """
        Placeholder to attach LoRA adapters to `model`.
        Wire this up later with PEFT or custom LoRA.
        """
        print(f"[LoRA] Placeholder active for {tag} (no adapters attached).")



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