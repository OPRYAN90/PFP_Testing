from typing import Any, Dict, Optional, List
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch.nn.functional as F
from .go_utils import build_go_index_from_dataset
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


class ProteinDataset(Dataset):
    """Dataset for multi‑modal protein function prediction.

    Loads per‑residue embeddings for ESM‑C, ProtT5, Ankh3‑XL, and XTrimoPGLM,
    and corresponding GO multi‑label targets.
    """

    # class-level cache so every worker only builds it once
    _go_dicts = {}

    def __init__(self, data_dir: str, protein_ids: List[str], task_type: str = "mf", split: str = "train"):
        """
        :param data_dir: Path to protein data directory
        :param protein_ids: List of protein IDs to include
        :param task_type: Which GO task(s) to load labels for ("mf", "bp", "cc")
        :param split: Which data split ("train", "val", "test")
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.task_type = task_type
        self.split = split
        # Sequences directory is automatically derived from data_dir
        self.sequences_dir = self.data_dir / "sequences"
        
        # CAFA_Style layout: <root>/<split>/<ontology>/<pid>
        self.split_dir = self.data_dir / split / task_type
        
    def __len__(self):
        return len(self.protein_ids)


    def _parse_go_labels(self, go_file_path: Path) -> torch.Tensor:
        """Parse GO IDs from ``<ontology>_go.txt`` into a multi‑hot tensor.

        Empty files yield an all‑zero vector. Unknown IDs are ignored with a warning.
        """
        # 1. fetch / cache mapping ------------------------------------------------
        if self.task_type not in self._go_dicts:
            # Build mapping from CAFA_Style training split
            self._go_dicts[self.task_type] = build_go_index_from_dataset(self.data_dir, self.task_type, split="train")
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
        protein_id = self.protein_ids[idx]
        # Directories for labels (by split) and for embeddings/sequences (global sequences dir)
        protein_dir = self.split_dir / protein_id
        seq_dir = self.sequences_dir / protein_id
        
        # Load sequence (from sequences_dir: seq.txt)
        with open(seq_dir / "seq.txt", "r") as f:
            sequence = f.read().strip()
            
        # Load pre-computed ESM-C embeddings (required file)
        esmc_file = seq_dir / "esmc_emb.pt"
        esmc_data = torch.load(esmc_file)
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor"
        esmc_emb = esmc_data.squeeze(0)
        
        # Load pre-computed Ankh3-XLarge embeddings (required file)
        ankh_file = seq_dir / "ankh_emb_xl.pt"
        ankh_data = torch.load(ankh_file)
        # Extract embeddings from data structures
        # Ankh3-XLarge: Direct tensor (1, L+2, d_ankh) -> squeeze to (L+2, d_ankh)
        assert ankh_data.dim() == 3, "Ankh3-XLarge data should be a 3D tensor"
        ankh_emb = ankh_data.squeeze(0)

        # Load pre-computed ProtT5 embeddings (required file)
        prot_file = seq_dir / "prot_t5_emb.pt"
        prot_data = torch.load(prot_file)
        assert prot_data.dim() == 3, "ProtT5 data should be a 3D tensor"
        prot_emb = prot_data.squeeze(0)

        # Load pre-computed XTrimoPGLM embeddings (required file)
        pglm_file = seq_dir / "pglm_emb.pt"
        pglm_data = torch.load(pglm_file)
        assert pglm_data.dim() == 3, "PGLM data should be a 3D tensor"
        pglm_emb = pglm_data.squeeze(0)

        # Ensure all embeddings have the same sequence length
        assert esmc_emb.size(0) == ankh_emb.size(0) == prot_emb.size(0) == pglm_emb.size(0), \
            f"Embeddings have different lengths: ESM-C={esmc_emb.size(0)}, Ankh-XLarge={ankh_emb.size(0)}, ProtT5={prot_emb.size(0)}, PGLM={pglm_emb.size(0)}"
            
        # Ensure that BOS/EOS token embeddings (positions 0 and L+1) are zeroed out for all models
        for emb in (esmc_emb, ankh_emb, prot_emb, pglm_emb):
            emb[0] = 0.0
            emb[-1] = 0.0

        # Load labels from CAFA_Style go.txt
        label_file = protein_dir / "go.txt"
        
        # Parse GO labels from text file format
        labels = self._parse_go_labels(label_file)
            
        # Protein length from sequence (no BOS/EOS)
        protein_length = len(sequence)
        
        sample = {
            "protein_id": protein_id,
            "sequence": sequence,
            "esmc_emb": esmc_emb,      # Shape: (L+2, d_esmc)
            "ankh_emb": ankh_emb,      # Shape: (L+2, d_ankh)
            "prot_emb": prot_emb,      # Shape: (L+2, d_prot)
            "pglm_emb": pglm_emb,      # Shape: (L+2, d_pglm)
            "length": protein_length,
            "labels": labels,
        }

        return sample


class ProteinDataModule(LightningDataModule):
    """DataModule for protein function prediction (no MSA).

    Loads precomputed per‑residue embeddings for ESM‑C, ProtT5, Ankh3‑XL, and XTrimoPGLM,
    and GO labels for ``train``, ``val``, and ``test`` splits.
    """

    # Help static type checkers recognise dynamically-added Hydra attribute
    hparams: Any

    def __init__(
        self,
        data_dir: str = "data",  
        task_type: str = "mf",  # "mf", "bp", or "cc"
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the data module.

        :param data_dir: Path to base directory.
        :param task_type: GO ontology to use ("mf", "bp", "cc").
        :param batch_size: Batch size per optimization step.
        :param num_workers: Number of data‑loading workers.
        :param pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        # Sequences directory is automatically derived from data_dir
        self.sequences_dir = Path(data_dir) / "sequences"
        
        # Store protein IDs for each split
        self.train_protein_ids: List[str] = []
        self.val_protein_ids: List[str] = []
        self.test_protein_ids: List[str] = []

    def get_num_classes(self, task_type_: str) -> int:
        """Return number of classes for the given ontology with validation.

        :param task_type_: Task type ("mf", "bp", "cc").
        :return: Number of classes for the task.
        """
        # Determine number of classes from CAFA_Style train split
        go_dict = build_go_index_from_dataset(Path(self.hparams.data_dir), task_type_, split="train")
        return len(go_dict)

    @property
    def num_classes(self) -> int:
        """Get number of classes for current task."""
        return self.get_num_classes(self.hparams.task_type)


    # ------------------------------------------------------------------
    #  GO-term mapping access (needed by ProteinLitModule._heal_metrics)
    # ------------------------------------------------------------------
    @property
    def _go_dicts(self) -> Dict[str, Dict[str, int]]:
        """Return the cached GO-term ➜ class-index mappings."""        
        return ProteinDataset._go_dicts

    def _find_valid_proteins_in_split(self, split_dir: Path) -> List[str]:
        """Scan a split directory to find proteins with required files."""
        valid_proteins = []
        
        print(f"Scanning {split_dir} for proteins with required files...")
        
        for protein_dir in split_dir.iterdir():
            if not protein_dir.is_dir():
                continue
                
            protein_id = protein_dir.name
            # Check that all required files exist
            required_files = [
                "seq.txt",
                "esmc_emb.pt",
                "ankh_emb_xl.pt",
                "prot_t5_emb.pt",
                "pglm_emb.pt",
                "go.txt"
            ]
            
            # Labels must be in split_dir; embeddings and seq are in sequences_dir
            seq_dir = self.sequences_dir / protein_id
            missing_files = []
            for f in required_files:
                if f == "go.txt":
                    if not (protein_dir / f).exists():
                        missing_files.append(f)
                else:
                    if not (seq_dir / f).exists():
                        missing_files.append(f)
            
            if missing_files:
                # This should not happen - all valid proteins must have these files
                raise FileNotFoundError(f"Protein {protein_id} missing required files: {missing_files}")
            
            valid_proteins.append(protein_id)
        
        print(f"Found {len(valid_proteins)} valid proteins in {split_dir}")
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
        
        # CAFA_Style split directories
        train_dir = data_dir / "train" / self.hparams.task_type
        val_dir = data_dir / "val" / self.hparams.task_type
        test_dir = data_dir / "test" / self.hparams.task_type
        
        missing_dirs = []
        if not train_dir.exists():
            missing_dirs.append(str(train_dir))
        if not val_dir.exists():
            missing_dirs.append(str(val_dir))
        if not test_dir.exists():
            missing_dirs.append(str(test_dir))
        
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

        # Ensure _go_dicts is populated for the current task_type in main process
        if self.hparams.task_type not in ProteinDataset._go_dicts:
            print(f"Populating GO mapping for '{self.hparams.task_type}' in main process...")
            ProteinDataset._go_dicts[self.hparams.task_type] = build_go_index_from_dataset(Path(self.hparams.data_dir), self.hparams.task_type, split="train")
            print(f"GO mapping loaded: {len(ProteinDataset._go_dicts[self.hparams.task_type])} terms")

        # Load datasets only if not already loaded
        if not self.data_train and not self.data_val and not self.data_test:
            
            print(f"Training proteins: {len(self.train_protein_ids)}")
            print(f"Validation proteins: {len(self.val_protein_ids)}")
            print(f"Test proteins: {len(self.test_protein_ids)}")
            
            # Create datasets for each split
            self.data_train = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.train_protein_ids,
                task_type=self.hparams.task_type,
                split="train",
            )
            
            self.data_val = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.val_protein_ids,
                task_type=self.hparams.task_type,
                split="val",
            )
            
            self.data_test = ProteinDataset(
                data_dir=self.hparams.data_dir,
                protein_ids=self.test_protein_ids,
                task_type=self.hparams.task_type,
                split="test",
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
    """Collate function for protein embeddings and labels.
    Pads per-residue embeddings to a common length and builds a boolean pad mask.
    """
    protein_ids = [it["protein_id"] for it in batch]
    sequences   = [it["sequence"]    for it in batch]
    lengths     = [it["length"]      for it in batch]   # residue counts (no CLS/EOS)

    # ----------------------------------------------------
    # 1) Pad sequence embeddings (float tensors)
    # ----------------------------------------------------
    max_len_seq = max(lengths) + 2   # CLS + residues + EOS
    # assert max_len_seq <= 1024, "Sequence too long (>1024)"

    esmc_emb_padded = []
    ankh_emb_padded = []
    prot_emb_padded = []
    pglm_emb_padded = []
    for it in batch:
        # Pad ESM-C embeddings
        esmc_emb = it["esmc_emb"]  # [L+2, d_esmc]
        if esmc_emb.size(0) < max_len_seq:
            esmc_emb = F.pad(esmc_emb, (0, 0, 0, max_len_seq - esmc_emb.size(0)), value=0)
        esmc_emb_padded.append(esmc_emb)
        
        # Pad Ankh3-XLarge embeddings (same padding logic)
        ankh_emb = it["ankh_emb"]  # [L+2, d_ankh]
        if ankh_emb.size(0) < max_len_seq:
            ankh_emb = F.pad(ankh_emb, (0, 0, 0, max_len_seq - ankh_emb.size(0)), value=0)
        ankh_emb_padded.append(ankh_emb)

        # Pad ProtT5 embeddings
        prot_emb = it["prot_emb"]  # [L+2, d_prot]
        if prot_emb.size(0) < max_len_seq:
            prot_emb = F.pad(prot_emb, (0, 0, 0, max_len_seq - prot_emb.size(0)), value=0)
        prot_emb_padded.append(prot_emb)

        # Pad PGLM embeddings
        pglm_emb = it["pglm_emb"]  # [L+2, d_pglm]
        if pglm_emb.size(0) < max_len_seq:
            pglm_emb = F.pad(pglm_emb, (0, 0, 0, max_len_seq - pglm_emb.size(0)), value=0)
        pglm_emb_padded.append(pglm_emb)

    # ----------------------------------------------------
    # 2) Stack embeddings + build masks
    # ----------------------------------------------------
    esmc_emb = torch.stack(esmc_emb_padded)       # [B, L_max_seq, d_esmc]
    ankh_emb = torch.stack(ankh_emb_padded)       # [B, L_max_seq, d_ankh]
    prot_emb = torch.stack(prot_emb_padded)       # [B, L_max_seq, d_prot]
    pglm_emb = torch.stack(pglm_emb_padded)       # [B, L_max_seq, d_pglm]

    labels = torch.stack([it["labels"] for it in batch])

    pad_mask = torch.tensor(
        [[i >= l + 2 for i in range(max_len_seq)] for l in lengths],
        dtype=torch.bool,
    )

    lengths_tensor = torch.tensor(lengths)

    return {
        "protein_id": protein_ids,
        "sequence": sequences,
        "esmc_emb": esmc_emb,
        "ankh_emb": ankh_emb,
        "prot_emb": prot_emb,
        "pglm_emb": pglm_emb,
        "labels": labels,
        "pad_mask": pad_mask,
        "lengths": lengths_tensor,
    }