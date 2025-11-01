from typing import Any, Dict, Optional, List
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch.nn.functional as F
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


# ======================================================================
#  Dataset (Regression) — Stability_Data/{train,val,test}/<protein>/...
#  Expected files in each protein dir:
#    - seq.txt            (AA sequence, plain text)
#    - label.txt          (single float, one line)
#    - esmc_emb.pt        (1, L+2, d)
#    - ankh_emb_xl.pt     (1, L+2, d)
#    - prot_t5_emb.pt     (1, L+2, d)
#    - pglm_emb.pt        (1, L+2, d)
# ======================================================================

class ProteinDataset(Dataset):
    """Dataset for regression on protein stability.

    Loads per-residue embeddings for ESM-C, ProtT5, Ankh3-XL, and XTrimoPGLM,
    plus a scalar target from label.txt.
    """

    def __init__(self, data_dir: str, protein_ids: List[str], split: str = "train"):
        """
        :param data_dir: Path to Stability_Data base directory
        :param protein_ids: List of protein directory names to include
        :param split: Which data split ("train", "val", "test")
        """
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.split = split

        # Split directory: .../Stability_Data/{train,val,test}
        self.split_dir = self.data_dir / split

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        protein_dir = self.split_dir / protein_id

        # Load sequence and compute length (no L.csv)
        seq_path = protein_dir / "seq.txt"
        sequence = seq_path.read_text(encoding="utf-8").strip()
        length_no_special = len(sequence)

        # Load pre-computed ESM-C embeddings (required)
        esmc_file = protein_dir / "esmc_emb.pt"
        esmc_data = torch.load(esmc_file)
        assert esmc_data.dim() == 3, "ESM-C data should be a 3D tensor (1, L+2, d)"
        esmc_emb = esmc_data.squeeze(0)  # (L+2, d)

        # Load pre-computed Ankh3-XL embeddings (required)
        ankh_file = protein_dir / "ankh_emb_xl.pt"
        ankh_data = torch.load(ankh_file)
        assert ankh_data.dim() == 3, "Ankh3-XL data should be a 3D tensor (1, L+2, d)"
        ankh_emb = ankh_data.squeeze(0)

        # Load pre-computed ProtT5 embeddings (required)
        prot_file = protein_dir / "prot_t5_emb.pt"
        prot_data = torch.load(prot_file)
        assert prot_data.dim() == 3, "ProtT5 data should be a 3D tensor (1, L+2, d)"
        prot_emb = prot_data.squeeze(0)

        # Load pre-computed XTrimoPGLM embeddings (required)
        pglm_file = protein_dir / "pglm_emb.pt"
        pglm_data = torch.load(pglm_file)
        assert pglm_data.dim() == 3, "PGLM data should be a 3D tensor (1, L+2, d)"
        pglm_emb = pglm_data.squeeze(0)

        # Ensure all embeddings share the same token length
        Lp2 = esmc_emb.size(0)
        assert (
            Lp2 == ankh_emb.size(0) == prot_emb.size(0) == pglm_emb.size(0)
        ), f"Embedding lengths differ: ESM-C={esmc_emb.size(0)}, Ankh={ankh_emb.size(0)}, ProtT5={prot_emb.size(0)}, PGLM={pglm_emb.size(0)}"

        # Zero BOS/EOS (positions 0 and L+1) if present
        for emb in (esmc_emb, ankh_emb, prot_emb, pglm_emb):
            emb[0] = 0.0
            emb[-1] = 0.0

        # Load scalar label
        label_path = protein_dir / "label.txt"
        label_str = label_path.read_text(encoding="utf-8").strip()
        label_val = float(label_str)
        label_tensor = torch.tensor(label_val, dtype=torch.float32)

        sample = {
            "protein_id": protein_id,
            "sequence": sequence,
            "esmc_emb": esmc_emb,      # (L+2, d_esmc)
            "ankh_emb": ankh_emb,      # (L+2, d_ankh)
            "prot_emb": prot_emb,      # (L+2, d_prot)
            "pglm_emb": pglm_emb,      # (L+2, d_pglm)
            "length": length_no_special,   # residues (no BOS/EOS)
            "labels": label_tensor,        # scalar float
        }
        return sample


# ======================================================================
#  DataModule (Regression)
# ======================================================================

class ProteinDataModule(LightningDataModule):
    """DataModule for protein regression (stability/fluorescence, no MSA)."""

    # Help static type checkers recognise dynamically-added Hydra attribute
    hparams: Any

    def __init__(
        self,
        data_dir: str = "/teamspace/studios/this_studio/PFP_Testing/data",
        task_type: str = "stability",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """
        :param data_dir: Either the dataset root (…/data) OR the specific task folder
                         (…/data/Stability_Data or …/data/Fluorescence_Data).
        :param task_type: 'stability' or 'fluorescence' (controls subfolder if root is given).
        :param batch_size: Batch size per optimization step.
        :param num_workers: Number of data-loading workers.
        :param pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # Store protein IDs for each split
        self.train_protein_ids: List[str] = []
        self.val_protein_ids: List[str] = []
        self.test_protein_ids: List[str] = []

    # -------------------------- split scanning --------------------------

    def _resolve_task_dir(self, base: Path) -> Path:
        """Return the concrete task directory, accepting either root or task folder."""
        # If user already passed the task folder, just use it.
        name = base.name.lower()
        if name.endswith("_data"):
            return base
        # Otherwise, append the task subfolder.
        sub = "Stability_Data" if self.hparams.task_type.lower().startswith("stab") else "Fluorescence_Data"
        return base / sub

    def _find_valid_proteins_in_split(self, split_dir: Path) -> List[str]:
        """Scan a split directory to find proteins with required files."""
        valid_proteins = []

        print(f"Scanning {split_dir} for regression samples...")

        required_files = [
            "seq.txt",
            "label.txt",
            "esmc_emb.pt",
            "ankh_emb_xl.pt",
            "prot_t5_emb.pt",
            "pglm_emb.pt",
        ]

        # Add rqdm progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, *a, **kw: x  # fallback in case tqdm is not installed

        protein_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        for protein_dir in tqdm(protein_dirs, desc=f"Scanning {split_dir.name} proteins"):
            missing = [f for f in required_files if not (protein_dir / f).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{protein_dir.name} missing required files: {missing}"
                )

            valid_proteins.append(protein_dir.name)

        print(f"Found {len(valid_proteins)} valid proteins in {split_dir.name}")
        return valid_proteins

    def prepare_data(self) -> None:
        """Find all proteins with required files in each split directory."""
        # Skip if already done
        if self.train_protein_ids and self.val_protein_ids and self.test_protein_ids:
            print(
                f"Using previously found proteins: "
                f"train={len(self.train_protein_ids)}, "
                f"val={len(self.val_protein_ids)}, "
                f"test={len(self.test_protein_ids)}"
            )
            return

        data_dir = Path(self.hparams.data_dir)
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        task_dir = self._resolve_task_dir(data_dir)

        # Split directories are exactly train/val/test
        train_dir = task_dir / "train"
        val_dir = task_dir / "val"
        test_dir = task_dir / "test"

        missing = [p.name for p in (train_dir, val_dir, test_dir) if not p.exists()]
        if missing:
            raise ValueError(f"Missing split directories in {data_dir}: {missing}")

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
        # Divide batch size by the number of devices, if applicable.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            print(f"Training proteins: {len(self.train_protein_ids)}")
            print(f"Validation proteins: {len(self.val_protein_ids)}")
            print(f"Test proteins: {len(self.test_protein_ids)}")

            base = str(self._resolve_task_dir(Path(self.hparams.data_dir)))
            self.data_train = ProteinDataset(
                data_dir=base, protein_ids=self.train_protein_ids, split="train"
            )
            self.data_val = ProteinDataset(
                data_dir=base, protein_ids=self.val_protein_ids, split="val"
            )
            self.data_test = ProteinDataset(
                data_dir=base, protein_ids=self.test_protein_ids, split="test"
            )

    def train_dataloader(self) -> DataLoader[Any]:
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
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {
            "train_protein_ids": self.train_protein_ids,
            "val_protein_ids": self.val_protein_ids,
            "test_protein_ids": self.test_protein_ids,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.train_protein_ids = state_dict.get("train_protein_ids", [])
        self.val_protein_ids = state_dict.get("val_protein_ids", [])
        self.test_protein_ids = state_dict.get("test_protein_ids", [])


# ======================================================================
#  Global collate function – unchanged except that labels are scalar
# ======================================================================

def protein_collate(batch):
    """Collate function for protein embeddings and labels.
    Pads per-residue embeddings to a common length and builds a boolean pad mask.
    """
    protein_ids = [it["protein_id"] for it in batch]
    sequences   = [it["sequence"]    for it in batch]
    lengths     = [it["length"]      for it in batch]   # residue counts (no CLS/EOS)

    # 1) Pad sequence embeddings (float tensors)
    max_len_seq = max(lengths) + 2   # CLS + residues + EOS
    esmc_emb_padded = []
    ankh_emb_padded = []
    prot_emb_padded = []
    pglm_emb_padded = []
    for it in batch:
        # ESM-C
        esmc_emb = it["esmc_emb"]  # [L+2, d_esmc]
        if esmc_emb.size(0) < max_len_seq:
            esmc_emb = F.pad(esmc_emb, (0, 0, 0, max_len_seq - esmc_emb.size(0)), value=0)
        esmc_emb_padded.append(esmc_emb.float())

        # Ankh3-XL
        ankh_emb = it["ankh_emb"]  # [L+2, d_ankh]
        if ankh_emb.size(0) < max_len_seq:
            ankh_emb = F.pad(ankh_emb, (0, 0, 0, max_len_seq - ankh_emb.size(0)), value=0)
        ankh_emb_padded.append(ankh_emb.float())

        # ProtT5
        prot_emb = it["prot_emb"]  # [L+2, d_prot]
        if prot_emb.size(0) < max_len_seq:
            prot_emb = F.pad(prot_emb, (0, 0, 0, max_len_seq - prot_emb.size(0)), value=0)
        prot_emb_padded.append(prot_emb.float())

        # PGLM
        pglm_emb = it["pglm_emb"]  # [L+2, d_pglm]
        if pglm_emb.size(0) < max_len_seq:
            pglm_emb = F.pad(pglm_emb, (0, 0, 0, max_len_seq - pglm_emb.size(0)), value=0)
        pglm_emb_padded.append(pglm_emb.float())

    # 2) Stack embeddings + build masks
    esmc_emb = torch.stack(esmc_emb_padded)       # [B, L_max_seq, d_esmc]
    ankh_emb = torch.stack(ankh_emb_padded)       # [B, L_max_seq, d_ankh]
    prot_emb = torch.stack(prot_emb_padded)       # [B, L_max_seq, d_prot]
    pglm_emb = torch.stack(pglm_emb_padded)       # [B, L_max_seq, d_pglm]

    # labels are scalar floats -> shape [B]
    labels = torch.stack([it["labels"] for it in batch]).float()

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
        "labels": labels,        # [B]
        "pad_mask": pad_mask,
        "lengths": lengths_tensor,
    }
