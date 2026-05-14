import hashlib
import numpy as np
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Set

# Try relative import, fallback to absolute
try:
    from .config import OCRConfig
except ImportError:
    from config import OCRConfig


def analyze_dataset(y: np.ndarray) -> Tuple[Counter, Dict[int, float], List[int]]:
    """
    Compute character frequencies, line weights for sampling, and rare character list.
    """
    cfg = OCRConfig()
    all_codes = y[y != 0]
    code_counts = Counter(all_codes)

    # Identify rare character codes (exclude space code 1)
    char_counts = {c: cnt for c, cnt in code_counts.items() if c != 1}
    rare_codes = [c for c, cnt in char_counts.items() if cnt < cfg.rare_threshold]

    # Compute line weights
    line_weights = {}
    for idx, seq in enumerate(y):
        seq_nonzero = seq[seq != 0]
        weight = 1.0
        if any(c in rare_codes for c in seq_nonzero):
            weight = cfg.rare_weight_multiplier
        line_weights[idx] = weight

    return code_counts, line_weights, rare_codes


def deduplicate_dataset(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Identify duplicate text lines and return indices to keep.
    """
    seq_hashes = {}
    duplicate_map = defaultdict(list)

    for idx, seq in enumerate(y):
        clean_seq = tuple(c for c in seq if c != 0)
        seq_hash = hashlib.md5(str(clean_seq).encode()).hexdigest()
        duplicate_map[seq_hash].append(idx)

    # Keep first occurrence of each hash
    keep_flags = np.ones(len(y), dtype=bool)
    for h, idxs in duplicate_map.items():
        for dup_idx in idxs[1:]:
            keep_flags[dup_idx] = False

    print(f"Deduplication: {len(y) - keep_flags.sum():,} duplicates removed, "
          f"{keep_flags.sum():,} unique samples remain.")

    return keep_flags, duplicate_map


def split_train_val_indices(y: np.ndarray, total_real_printed: int, val_split: float,
                            keep_mask: np.ndarray, duplicate_map: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split real printed subset into train/val, ensuring no duplicate sequences cross the split.
    """
    # Real printed indices are 0 to total_real_printed-1
    real_indices = np.arange(total_real_printed)
    # Apply keep_mask to avoid duplicates
    real_keep = real_indices[keep_mask[real_indices]]

    # Group by hash to keep duplicates together
    hash_to_real_indices = defaultdict(list)
    for idx in real_keep:
        clean_seq = tuple(c for c in y[idx] if c != 0)
        h = hashlib.md5(str(clean_seq).encode()).hexdigest()
        hash_to_real_indices[h].append(idx)

    # Shuffle hashes and split
    unique_hashes = list(hash_to_real_indices.keys())
    np.random.seed(42)
    np.random.shuffle(unique_hashes)

    n_val_hashes = int(len(unique_hashes) * val_split)
    val_hashes = set(unique_hashes[:n_val_hashes])

    val_indices = []
    train_indices = []
    for h in unique_hashes:
        idxs = hash_to_real_indices[h]
        if h in val_hashes:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    # Convert to boolean masks
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True

    print(f"Train real printed samples: {len(train_indices):,}")
    print(f"Val real printed samples: {len(val_indices):,}")
    return train_mask, val_mask
