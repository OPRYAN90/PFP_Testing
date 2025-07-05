from typing import Any, Dict, Optional, Tuple, List
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import get_worker_info
import pickle as pkl
from .go_utils import load_go_dict
from torch._dynamo import OptimizedModule      
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

class ProteinDataset(Dataset):
    """Custom Dataset for multi-modal protein function prediction."""
    
    # ── GLOBAL CACHES (per-process, not per-thread) ────────────────
    _msa_model         = None
    _msa_batch_conv    = None

    # static device decision: *this* process only
    _device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _use_bf16          = ( #NOTE: CHANGE GPU PRECISION
        _device.type == "cuda" and torch.cuda.get_device_capability(_device)[0] >= 8
    )
    
    # class-level cache so every worker only builds it once
    _go_dicts = {}
    
    def __init__(self, data_dir: str, protein_ids: List[str], task_type: str = "mf", split: str = "train", compile_msa_model: bool = True):
        """
        :param data_dir: Path to protein data directory (base PDBCH directory)
        :param protein_ids: List of protein IDs to include
        :param task_type: Which GO task(s) to load labels for ("mf", "bp", "cc")
        :param split: Which data split ("train", "val", "test")
        :param compile_msa_model: Whether to compile ESM-MSA model for speed (default: True)
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.task_type = task_type
        self.split = split
        self.compile_msa_model = compile_msa_model
        
        # Determine the specific split directory
        self.split_dir = self.data_dir / f"{split}_pdbch"
        
    def __len__(self):
        return len(self.protein_ids)
    
    # --------------------------------------------------------------
    @classmethod
    def _get_msa_model(cls, compile_model: bool = True):
        """
        Load the ESM-MSA model once per process with optional torch.compile optimization.
        
        :param compile_model: Whether to compile the model for speed optimization
        """
        
        if cls._msa_model:
            return cls._msa_model, cls._msa_batch_conv

        try:
            import esm
            model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            cls._msa_batch_conv = alphabet.get_batch_converter()
            
            # Move to device and keep master weights in FP32
            model.eval().to(cls._device)
            
            # Conditionally compile the model for significant speed improvements
            if compile_model:
                try:
                    if get_worker_info() is None:  # main process only
                        print(f"🚀 Compiling ESM-MSA model with torch.compile...")
                    
                    cls._msa_model = torch.compile(
                        model, 
                        mode='default',  
                       dynamic=True            
                    )
                    
                    if get_worker_info() is None:
                        print(f"✅ ESM-MSA model compiled successfully")
                        
                except Exception as compile_error:
                    # Fallback to uncompiled model if compilation fails
                    if get_worker_info() is None:
                        print(f"⚠️  Compilation failed ({compile_error}), using uncompiled model")
                    cls._msa_model = model
            else:
                cls._msa_model = model
                if get_worker_info() is None:
                    print(f"📋 ESM-MSA model loaded without compilation (as requested)")
                
            if get_worker_info() is None:        # main process only
                is_compiled = isinstance(cls._msa_model, OptimizedModule)
                print(
                    f"[ESM-MSA] loaded on {cls._device} "
                    f"({'BF16' if cls._use_bf16 else 'FP32'}) "
                    f"{'[COMPILED]' if is_compiled else '[EAGER]'}"
                )
                
        except Exception as e:
            raise RuntimeError(f"ESM-MSA load failed: {e}")

        return cls._msa_model, cls._msa_batch_conv

    @staticmethod
    def _parse_a3m(path: Path) -> List[str]:
        """Parse A3M file - first sequence is query, max 256 total sequences."""
        seqs, seq = [], []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                if line[0] == ">":                     # header
                    if seq:
                        seqs.append("".join(seq))
                    seq = []                           # reset
                else:
                    seq.append(line)
            if seq:  # add final sequence
                seqs.append("".join(seq))
        
        assert 1 <= len(seqs) <= 256, f"Expected 1-256 sequences, got {len(seqs)} from {path}"
        return seqs
    #TODO: ASSERT ESM-C PROPER BF16 # and test bf model-wise
    @torch.inference_mode()
    def _compute_msa_embeddings(self, a3m_file: Path) -> Tuple[torch.Tensor, float, float]:
        t_parse = time.perf_counter()

        model, batch_converter = self._get_msa_model(compile_model=self.compile_msa_model)
        
        # Time A3M parsing
        seqs = self._parse_a3m(a3m_file)
        
        msa = [(f"seq{i}", s) for i, s in enumerate(seqs)]

        _, _, tok = batch_converter([msa])  # type: ignore[arg-type]
        tok = tok.to(self._device)
        a3m_parse_time = time.perf_counter() - t_parse

        # Time actual model computation
        _t_compute = time.perf_counter()
        # Optimized autocast context for compiled models
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._use_bf16,
        ):
            # For compiled models, this will be significantly faster
            rep = model(tok, repr_layers=[12])["representations"][12]
        
        embeddings = rep.to(torch.float32).squeeze(0).cpu()  # (N_seq, L+1, d) - keep .cpu() to avoid memory issues
        msa_compute_time = time.perf_counter() - _t_compute
        return embeddings, a3m_parse_time, msa_compute_time
        #NOTE: COMPUTATION NOT BATCH_PARALLELIZED
    def _parse_go_labels(self, go_file_path: Path) -> torch.Tensor:
        """
        Convert the GO IDs inside <ontology>_go.txt into a multi-label tensor.
        Empty file  ➜  all-zero vector (negative example).
        Unknown ID ➜  ignored but logged once.
        """
        # 1. fetch / cache mapping ------------------------------------------------
        if self.task_type not in self._go_dicts:
            # TSV file is in the parent directory of data_dir (same as in get_num_classes)
            tsv_path = self.data_dir.parent / "nrPDB-GO_2019.06.18_annot.tsv"
            self._go_dicts[self.task_type] = load_go_dict(tsv_path, self.task_type)
        go2idx = self._go_dicts[self.task_type]
        num_classes = len(go2idx)

        # 2. build empty vector ---------------------------------------------------
        labels = torch.zeros(num_classes, dtype=torch.float32)

        # 3. fill in positive indices -------------------------------------------
        if go_file_path.exists() and go_file_path.stat().st_size > 0:
            with go_file_path.open() as fh:
                for raw in fh:
                    go_id = raw.strip()
                    if not go_id:
                        continue
                    try:
                        labels[go2idx[go_id]] = 1.0
                    except KeyError:
                        print(f"[WARN] {go_id} not in mapping ({self.task_type})")
        return labels
    
    def __getitem__(self, idx):
        # Measure full sample preparation time
        _t0 = time.perf_counter()
        
        protein_id = self.protein_ids[idx]
        protein_dir = self.split_dir / protein_id
        
        # Load sequence
        with open(protein_dir / "sequence.txt", "r") as f:
            sequence = f.read().strip()
            
        # Load pre-computed ESM-C embeddings (required file)
        esmc_file = protein_dir / "esmc_emb.pt"
        esmc_data = torch.load(esmc_file)
        # Extract embeddings from data structures
        # ESM-C: Direct tensor (1, L+2, 1152) -> squeeze to (L+2, 1152)
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor"
        esmc_emb = esmc_data.squeeze(0)
        #track this starting here 
        # ------------------------------------------------------------------
        # Measure time spent **inside** the expensive MSA embedding routine
        # ------------------------------------------------------------------
        _t_msa = time.perf_counter()
        # Load MSA data from .a3m file and compute embeddings on-the-fly
        a3m_file = protein_dir / "final_filtered_256_stripped.a3m"
        msa_emb, a3m_parse_time, msa_compute_time = self._compute_msa_embeddings(a3m_file)
        msa_time = time.perf_counter() - _t_msa
        #track this ending here 
        # Load labels based on task type from GO text files
        label_file = protein_dir / f"{self.task_type}_go.txt"
        
            # Parse GO labels from text file format
        labels = self._parse_go_labels(label_file)
            
        # Load protein length info (L.csv just contains a single number on the first line)
        with open(protein_dir / "L.csv", "r") as f:
            protein_length = int(f.readline().strip())
        
        assert protein_length == esmc_emb.size(0)-2 == msa_emb.size(1)-1, "Sequence/MSA/Length mismatch"
        
        sample = {
            "protein_id": protein_id,
            "sequence": sequence,
            "sequence_emb": esmc_emb,  # Shape: (L+2, d_model) 
            "msa_emb": msa_emb,        # Shape: (N_seq, L+1, d_msa)
            "labels": labels,
            "length": protein_length,
            # expose per-sample timing of the MSA embedding step so that
            # the collate function (and perf callback) can aggregate it
            "msa_time": msa_time,
            "a3m_parse_time": a3m_parse_time,
            "msa_compute_time": msa_compute_time,
        }

        # attach timing (seconds) for this sample
        sample["sample_prep_time"] = time.perf_counter() - _t0

        return sample


class ProteinDataModule(LightningDataModule):
    """DataModule for protein function prediction.
    
    Handles two input modalities:
    1. ESM-C embeddings (pre-computed, required as .pt files)
    2. MSA embeddings (computed on-the-fly from .a3m files)
    
    Now supports three predefined splits: train, val, test from separate directories.
    """

    # Help static type checkers recognise dynamically-added Hydra attribute
    hparams: Any

    def __init__(
        self,
        data_dir: str = "data/PDBCH",  # Updated to point to PDBCH base directory
        task_type: str = "mf",  # "mf", "bp", or "cc"
        batch_size: int = 4,    # Updated to 4 as requested
        num_workers: int = 0,
        pin_memory: bool = True,
        compile_msa_model: bool = True,  # Whether to compile ESM-MSA model
    ) -> None:
        """Initialize ProteinDataModule.

        :param data_dir: Path to PDBCH base directory 
        :param task_type: Which GO tasks to train on ("mf", "bp", "cc")
        :param batch_size: Batch size for training
        :param num_workers: Number of data loading workers
        :param pin_memory: Whether to pin memory for GPU transfer
        :param compile_msa_model: Whether to compile ESM-MSA model for speed
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        # Store protein IDs for each split
        self.train_protein_ids: List[str] = []
        self.val_protein_ids: List[str] = []
        self.test_protein_ids: List[str] = []

    def get_num_classes(self, task_type_: str) -> int:
        """Get number of classes for a given task type with validation.
        
        :param task_type: Task type ("mf", "bp", "cc")
        :return: Number of classes for the task
        """
        expected_counts = {"mf": 489, "bp": 1943, "cc": 320}
        assert task_type_ in expected_counts, f"Unknown task_type: {task_type_}"
        
        # Look for TSV file in parent directory of data_dir
        data_base_dir = Path(self.hparams.data_dir).parent
        tsv = data_base_dir / "nrPDB-GO_2019.06.18_annot.tsv"
        go_dict = load_go_dict(tsv, task_type_)
        actual_count = len(go_dict)
        
        assert actual_count == expected_counts[task_type_], \
            f"Expected {expected_counts[task_type_]} classes for {task_type_}, got {actual_count}"
        
        return actual_count

    @property
    def num_classes(self) -> int:
        """Get number of classes for current task."""
        return self.get_num_classes(self.hparams.task_type)

    @property
    def ic_vector(self) -> torch.Tensor:
        """Get the information content vector for Smin computation."""
        return self._ic_vector

    # ------------------------------------------------------------------
    #  GO-term mapping access (needed by ProteinLitModule._heal_metrics)
    # ------------------------------------------------------------------
    @property
    def _go_dicts(self) -> Dict[str, Dict[str, int]]:
        """Return the cached GO-term ➜ class-index mappings."""        
        return ProteinDataset._go_dicts

    def _find_valid_proteins_in_split(self, split_dir: Path) -> List[str]:
        """Scan a split directory to find proteins with non-empty GO label files."""
        valid_proteins = []
        
        print(f"Scanning {split_dir} for proteins with non-empty {self.hparams.task_type} labels...")
        
        for protein_dir in split_dir.iterdir():
            if not protein_dir.is_dir():
                continue
                
            protein_id = protein_dir.name
            # Check that all required files exist
            required_files = [
                "L.csv",
                "sequence.txt", 
                "final_filtered_256_stripped.a3m",
                "esmc_emb.pt",
                f"{self.hparams.task_type}_go.txt"
            ]
            
            missing_files = [f for f in required_files if not (protein_dir / f).exists()]
            
            if missing_files:
                # This should not happen - all valid proteins must have these files
                raise FileNotFoundError(f"Protein {protein_id} missing required files: {missing_files}")
            
            valid_proteins.append(protein_id)
        
        print(f"Found {len(valid_proteins)} valid proteins with {self.hparams.task_type} labels in {split_dir.name}")
        return valid_proteins

    def prepare_data(self) -> None:
        """Find all proteins with required files in each split directory."""
        # Skip if already done
        if self.train_protein_ids and self.val_protein_ids and self.test_protein_ids:
            print(f"Using previously found proteins: "
                  f"train={len(self.train_protein_ids)}, "
                  f"val={len(self.val_protein_ids)}, "
                  f"test={len(self.test_protein_ids)}")
            return
        
        data_dir = Path(self.hparams.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Check for all three split directories
        train_dir = data_dir / "train_pdbch"
        val_dir = data_dir / "val_pdbch"
        test_dir = data_dir / "test_pdbch"
        
        missing_dirs = []
        if not train_dir.exists():
            missing_dirs.append("train_pdbch")
        if not val_dir.exists():
            missing_dirs.append("val_pdbch")
        if not test_dir.exists():
            missing_dirs.append("test_pdbch")
        
        if missing_dirs:
            raise ValueError(f"Missing split directories in {data_dir}: {missing_dirs}")
        
        # Find valid proteins in each split
        self.train_protein_ids = self._find_valid_proteins_in_split(train_dir)
        self.val_protein_ids = self._find_valid_proteins_in_split(val_dir)
        self.test_protein_ids = self._find_valid_proteins_in_split(test_dir)
        
        if len(self.train_protein_ids) == 0:
            raise ValueError("No valid training proteins found")
        if len(self.val_protein_ids) == 0:
            raise ValueError("No valid validation proteins found")
        if len(self.test_protein_ids) == 0:
            raise ValueError("No valid test proteins found")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for each split."""
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets only if not already loaded
        if not self.data_train and not self.data_val and not self.data_test:
            
            print(f"Training proteins ({len(self.train_protein_ids)}): {self.train_protein_ids[:5]}{'...' if len(self.train_protein_ids) > 5 else ''}")
            print(f"Validation proteins ({len(self.val_protein_ids)}): {self.val_protein_ids[:5]}{'...' if len(self.val_protein_ids) > 5 else ''}")
            print(f"Test proteins ({len(self.test_protein_ids)}): {self.test_protein_ids[:5]}{'...' if len(self.test_protein_ids) > 5 else ''}")
            
            # Create datasets for each split
            self.data_train = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.train_protein_ids,
                task_type=self.hparams.task_type,
                split="train",
                compile_msa_model=self.hparams.compile_msa_model
            )
            
            # ---------------------------------------------------------
            # HEAL information–content vector – loaded from ic_count.pkl
            # ---------------------------------------------------------
            if not hasattr(self, "_ic_vector"):
                # Use the parent directory of data_dir to find ic_count.pkl
                data_base_dir = Path(self.hparams.data_dir).parent
                ic_file = data_base_dir / "ic_count.pkl"
                if ic_file.exists():
                    with ic_file.open("rb") as f:
                        ic_count = pkl.load(f)               # dict with keys 'bp','mf','cc'
                    counts = torch.tensor(
                        ic_count[self.hparams.task_type], dtype=torch.float
                    )
                    counts[counts == 0] = 1                  # avoid log(0)
                    # HEAL constant: 69 709 training proteins
                    self._ic_vector = (-torch.log2(counts / 69_709)).float()
                    print(f"✅  IC vector loaded from {ic_file}")
                else:
                    raise FileNotFoundError(f"ic_count.pkl not found at {ic_file}")
            # Verify length agreement
            assert len(self._ic_vector) == self.get_num_classes(self.hparams.task_type)
            
            self.data_val = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.val_protein_ids,
                task_type=self.hparams.task_type,
                split="val",
                compile_msa_model=self.hparams.compile_msa_model
            )
            
            self.data_test = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.test_protein_ids,
                task_type=self.hparams.task_type,
                split="test",
                compile_msa_model=self.hparams.compile_msa_model
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=protein_collate,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up resources."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint."""
        return {
            "train_protein_ids": self.train_protein_ids,
            "val_protein_ids": self.val_protein_ids,
            "test_protein_ids": self.test_protein_ids,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self.train_protein_ids = state_dict.get("train_protein_ids", [])
        self.val_protein_ids = state_dict.get("val_protein_ids", [])
        self.test_protein_ids = state_dict.get("test_protein_ids", [])

# ======================================================================
#  Global collate function – picklable for multiprocessing 'spawn'
# ======================================================================

def protein_collate(batch):
    """Custom collate function to handle variable-length sequences and matrices.

    This is identical to ``ProteinDataModule._collate_fn`` but defined at
    module scope so that it can be pickled when DataLoader workers are
    started with the *spawn* start-method.  It relies only on ``torch`` and
    ``torch.nn.functional`` (aliased ``F``) which are already imported at the
    top of this file.
    """
    t0 = time.perf_counter()  # start collate timing

    protein_ids = [item["protein_id"] for item in batch]
    sequences   = [item["sequence"] for item in batch]
    lengths     = [item["length"]   for item in batch]        # residue counts

    # +2  (CLS + EOS)  — every sequence tensor on disk already has them
    # +2  (CLS + EOS)  — every MSA row on disk has CLS; EOS is added later
    max_len_seq = max(lengths) + 2        # unchanged (CLS+res+EOS)
    assert max_len_seq <= 1024, "Sequence length must be less than or equal to 1024"

    # Pad sequence embeddings  [max_len_seq, d_model]
    seq_embs = []
    for item in batch:
        seq_emb = item["sequence_emb"]  # [L, d_model]
        if seq_emb.size(0) < max_len_seq:
            pad_size = max_len_seq - seq_emb.size(0)
            seq_emb = F.pad(seq_emb, (0, 0, 0, pad_size), value=0)
        seq_embs.append(seq_emb)

    # ── NEW: row-axis padding ────────────────────────────────────
    max_n_seq = max(item["msa_emb"].size(0) for item in batch)

    msa_embs = []
    for item in batch:
        msa = item["msa_emb"]           # [N_seq, L, d_msa]

        # pad L axis (existing code)
        if msa.size(1) < max_len_seq:
            pad_L = max_len_seq - msa.size(1)
            msa = F.pad(msa, (0, 0, 0, pad_L, 0, 0), value=0)

        # pad N_seq axis (row axis) **at the bottom**
        if msa.size(0) < max_n_seq:
            pad_rows = max_n_seq - msa.size(0)
            msa = F.pad(msa, (0, 0, 0, 0, 0, pad_rows), value=0)

        msa_embs.append(msa)

    # Stack embeddings
    sequence_emb = torch.stack(seq_embs)    # [B, L_seq_max, d_model]
    msa_emb      = torch.stack(msa_embs)    # [B, N_seq, L_msa_max, d_msa]

    assert sequence_emb.size(1) == msa_emb.size(2), "Sequence and MSA lengths must be the same"

    # Stack labels
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    # Masking boolean tensor [B, max_len_seq]
    pad_mask = torch.tensor(
        [[i >= l + 2 for i in range(max_len_seq)] for l in lengths],
        dtype=torch.bool
    )

    # Build lengths tensor 
    lengths = torch.tensor(lengths)

    # ------------------------------------------------------------------
    # Aggregate per-sample preparation time (from Dataset) + collate time
    # ------------------------------------------------------------------
    sample_prep_total = 0.0
    msa_time_total = 0.0
    a3m_parse_time_total = 0.0
    msa_compute_time_total = 0.0
    for item in batch:
        sample_prep_total += float(item.get("sample_prep_time", 0.0))
        msa_time_total   += float(item.get("msa_time", 0.0))
        a3m_parse_time_total += float(item.get("a3m_parse_time", 0.0))
        msa_compute_time_total += float(item.get("msa_compute_time", 0.0))

    collate_time = time.perf_counter() - t0
    batch_dict = {
        "protein_id": protein_ids,
        "sequence": sequences,
        "sequence_emb": sequence_emb,
        "msa_emb": msa_emb,
        "labels": labels,
        "pad_mask": pad_mask,
        "lengths": lengths,
        "prep_time": sample_prep_total + collate_time,
        "msa_time": msa_time_total,
        "a3m_parse_time": a3m_parse_time_total,
        "msa_compute_time": msa_compute_time_total,
    }

    return batch_dict