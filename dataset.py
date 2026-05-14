import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from typing import Optional, Dict, List, Tuple
import json
from collections import Counter

# Use relative imports
try:
    from .config import OCRConfig
    from .preprocessing import preprocess_image, decode_label_sequence
    from .augmentation import get_augmentation_pipeline
    from .utils import analyze_dataset, deduplicate_dataset
except ImportError:
    from config import OCRConfig
    from preprocessing import preprocess_image, decode_label_sequence
    from augmentation import get_augmentation_pipeline
    from utils import analyze_dataset, deduplicate_dataset

# ... rest of dataset.py remains the same ...

# ... rest of the dataset.py code remains the same ...

class AmharicOCRDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 indices: np.ndarray,                # Boolean mask or list of indices to use
                 char_to_idx: Dict[int, int],
                 line_weights: Optional[Dict[int, float]] = None,
                 augment: bool = False,
                 cfg: OCRConfig = OCRConfig()):
        """
        Args:
            X, y: Full dataset arrays (memory-mapped).
            indices: Boolean mask or integer indices of samples to include.
            char_to_idx: Mapping from code (1-279) to model class index (0-278).
                         (CTC blank will be index 279).
            line_weights: Optional per-sample weights for weighted sampling.
            augment: Whether to apply data augmentation.
            cfg: Configuration object.
        """
        self.X = X
        self.y = y
        if indices.dtype == bool:
            self.indices = np.where(indices)[0]
        else:
            self.indices = indices
        self.char_to_idx = char_to_idx
        self.line_weights = line_weights
        self.augment = augment
        self.cfg = cfg

        if augment:
            self.aug_pipeline = get_augmentation_pipeline(
                rotation_deg=cfg.rotation_degrees,
                elastic_alpha=cfg.elastic_alpha,
                elastic_sigma=cfg.elastic_sigma,
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit
            )
        else:
            self.aug_pipeline = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        img = self.X[global_idx].copy()
        label_codes = self.y[global_idx]

        img_proc = preprocess_image(
            img,
            target_height=self.cfg.img_height,
            target_width=self.cfg.img_width,
            normalize_max=self.cfg.normalize_max,
            augment_fn=None   # Augmentation disabled
        )

        label_indices = decode_label_sequence(label_codes, self.char_to_idx)
        return torch.from_numpy(img_proc), torch.from_numpy(label_indices)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler using the provided line_weights."""
        if self.line_weights is None:
            raise ValueError("line_weights must be provided to use weighted sampling.")
        weights = [self.line_weights[global_idx] for global_idx in self.indices]
        sampler = WeightedRandomSampler(weights, num_samples=len(self), replacement=True)
        return sampler

def build_char_mappings(code_counts: Counter) -> Tuple[Dict[int, int], List[int]]:
    """
    Create mapping from dataset codes to contiguous class indices (0-based).
    Also return list of codes in order of class index (for inverse mapping).
    CTC blank will be appended as the last class.
    """
    # Sort codes by frequency (descending) for potential efficiency
    sorted_codes = [code for code, _ in code_counts.most_common()]
    # Ensure space (code 1) is included and given a consistent index
    if 1 not in sorted_codes:
        sorted_codes.insert(0, 1)

    char_to_idx = {code: i for i, code in enumerate(sorted_codes)}
    idx_to_char = sorted_codes
    print(f"Created mapping for {len(char_to_idx)} classes (including space).")
    return char_to_idx, idx_to_char

def create_datasets(cfg: OCRConfig) -> Tuple[Dataset, Dataset, Dict]:
    """
    Load data, deduplicate, split, and create train/val datasets.
    Returns train_dataset, val_dataset, and a dict with metadata.
    """
    from amharic_ocr.utils import analyze_dataset, deduplicate_dataset, split_train_val_indices

    # Load data (memory-mapped)
    X_path = os.path.join(cfg.data_dir, 'X_trainp_pg_vg.npy')
    y_path = os.path.join(cfg.data_dir, 'y_trainp_pg_vg.npy')
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')

    # Total real printed count from documentation
    N_REAL = 40929

    # 1. Deduplicate (across whole dataset)
    keep_mask, duplicate_map = deduplicate_dataset(y)

    # 2. Analyze rare characters and compute line weights
    code_counts, line_weights, rare_codes = analyze_dataset(y)
    print(f"Rare characters (≤{cfg.rare_threshold} occurrences): {len(rare_codes)}")

    # 3. Split real printed into train/val, respecting duplicates
    train_mask, val_mask = split_train_val_indices(y, N_REAL, cfg.val_split, keep_mask, duplicate_map)

    # 4. For training, we can optionally include synthetic data (deduplicated)
    # Here we'll use only real printed + a portion of deduplicated synthetic for training.
    # (You can adjust based on your computational resources)
    synthetic_keep = keep_mask[N_REAL:] & ~val_mask[N_REAL:]   # exclude val area entirely
    # We'll use up to 100k synthetic samples to keep training manageable
    synthetic_indices = np.where(synthetic_keep)[0] + N_REAL
    if len(synthetic_indices) > 100000:
        np.random.seed(42)
        synthetic_indices = np.random.choice(synthetic_indices, 100000, replace=False)
    train_indices = np.concatenate([np.where(train_mask)[0], synthetic_indices])

    # Validation uses only real printed (deduplicated and split)
    val_indices = np.where(val_mask)[0]

    # 5. Build character mappings
    char_to_idx, idx_to_char = build_char_mappings(code_counts)

    # 6. Create datasets
    train_dataset = AmharicOCRDataset(
        X, y, indices=train_indices,
        char_to_idx=char_to_idx,
        line_weights=line_weights,
        augment=cfg.use_augmentation,
        cfg=cfg
    )
    val_dataset = AmharicOCRDataset(
        X, y, indices=val_indices,
        char_to_idx=char_to_idx,
        line_weights=None,
        augment=False,
        cfg=cfg
    )

    metadata = {
        'code_counts': code_counts,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'rare_codes': rare_codes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'duplicate_map': duplicate_map
    }

    return train_dataset, val_dataset, metadata
