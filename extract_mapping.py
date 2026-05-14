#!/usr/bin/env python3
# extract_mapping.py – save the char_to_idx mapping to a JSON file
import numpy as np
import json, os, sys
from collections import Counter

# Path to your training labels
DATA_DIR = os.path.expanduser("~/Desktop/vscode/C++_NLP/amharicCV/train_big/train")
y_path = os.path.join(DATA_DIR, "y_trainp_pg_vg.npy")

if not os.path.exists(y_path):
    print(f"ERROR: Cannot find {y_path}")
    sys.exit(1)

print("Loading labels ...")
y = np.load(y_path, mmap_mode='r')
all_codes = y[y != 0]
code_counts = Counter(all_codes)

# Build mapping (same logic as in your training)
sorted_codes = [code for code, _ in code_counts.most_common()]
if 1 not in sorted_codes:          # ensure space (code 1) is present
    sorted_codes.insert(0, 1)

char_to_idx = {int(code): i for i, code in enumerate(sorted_codes)}
idx_to_char = {i: int(code) for code, i in char_to_idx.items()}

metadata = {
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "num_classes": len(char_to_idx) + 1,   # +1 for CTC blank
    "img_height": 64,
    "img_width": 256,
    "rnn_hidden_size": 256
}

output_path = "metadata.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Metadata saved to {output_path}")
print(f"   Number of classes (incl. blank): {metadata['num_classes']}")
