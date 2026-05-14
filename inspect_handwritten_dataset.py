#!/usr/bin/env python3
"""
inspect_handwritten_dataset.py – Deep inspection of the Amharic Handwritten dataset.
Analyzes: shapes, splits, pixel stats, label distributions, handwriting vs augmented,
image quality, label lengths, character frequencies, and visual samples.

Usage:
    python inspect_handwritten_dataset.py --data_dir ./npy64by256 --output_dir ./handwritten_inspection
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import json
import textwrap

# Optional nice plots
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
def load_data(data_dir):
    """Load all .npy files from the 64x256 handwritten dataset."""
    files = {}
    required = [
        'X_train.npy', 'y_train.npy', 'train_input_length.npy', 'train_label_length.npy',
        'X_val.npy',   'y_val.npy',   'valid_input_length.npy', 'valid_label_length.npy',
        'X_test.npy',  'y_test.npy',  'test_input_length.npy',  'test_label_length.npy'
    ]
    
    for fname in required:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")
        key = fname.replace('.npy','')
        files[key] = np.load(fpath, mmap_mode='r')
        print(f"  ✓ Loaded {fname}: {files[key].shape} ({files[key].dtype})")
    
    # Fix validation key naming
    files['val_input_length'] = files.pop('valid_input_length')
    files['val_label_length'] = files.pop('valid_label_length')
    
    return files


# -----------------------------------------------------------------------------
# 1. Basic statistics
# -----------------------------------------------------------------------------
def basic_statistics(files, output_dir, report):
    """Print and save basic array information."""
    report.append("## 1. Dataset Overview\n")
    report.append("| Split | Images | Shape | Dtype | Input Len Shape | Label Len Shape |")
    report.append("|-------|--------|-------|-------|-----------------|-----------------|")
    
    for split in ['train', 'val', 'test']:
        X = files[f'X_{split}']
        y = files[f'y_{split}']
        il = files[f'{split}_input_length']
        ll = files[f'{split}_label_length']
        
        report.append(f"| {split.upper():5s} | {X.shape[0]:6,} | {X.shape[1:]} | {X.dtype} | {il.shape} | {ll.shape} |")
    
    total_images = files['X_train'].shape[0] + files['X_val'].shape[0] + files['X_test'].shape[0]
    report.append(f"\n**Total images:** {total_images:,}")
    report.append(f"**Image dimensions:** {files['X_train'].shape[1]} (H) × {files['X_train'].shape[2]} (W)")
    report.append("")


# -----------------------------------------------------------------------------
# 2. Pixel statistics per split
# -----------------------------------------------------------------------------
def pixel_statistics(files, output_dir, report):
    """Analyze pixel intensity distributions per split."""
    report.append("## 2. Pixel Intensity Analysis\n")
    report.append("| Split | Min | Max | Mean ± Std |")
    report.append("|-------|-----|-----|------------|")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {'train': 'blue', 'val': 'green', 'test': 'orange'}
    
    for ax, split in zip(axes, ['train', 'val', 'test']):
        X = files[f'X_{split}']
        # Sample to avoid memory issues
        sample_size = min(5000, X.shape[0])
        sample = X[np.random.choice(X.shape[0], sample_size, replace=False)].ravel()
        
        mean_val = sample.mean()
        std_val = sample.std()
        min_val = sample.min()
        max_val = sample.max()
        
        report.append(f"| {split.upper():5s} | {min_val:.4f} | {max_val:.4f} | {mean_val:.4f} ± {std_val:.4f} |")
        
        ax.hist(sample, bins=50, alpha=0.7, color=colors[split], edgecolor='black')
        ax.set_title(f'{split.upper()} Pixel Distribution')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.suptitle('Pixel Intensity Distribution by Split', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_distribution.png'), dpi=150)
    plt.close()
    report.append(f"\n![Pixel distributions](pixel_distribution.png)\n")


# -----------------------------------------------------------------------------
# 3. Label analysis
# -----------------------------------------------------------------------------
def analyze_labels(files, output_dir, report):
    """Analyze label lengths and character frequencies."""
    report.append("## 3. Label Analysis\n")
    
    # Combine all labels for character analysis
    all_labels = []
    label_lengths = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        y = files[f'y_{split}']
        ll = files[f'{split}_label_length']
        
        for i in range(y.shape[0]):
            seq = y[i]
            # Remove padding (0)
            nonzero = seq[seq != 0]
            all_labels.extend(nonzero.tolist())
            label_lengths[split].append(len(nonzero))
    
    # Label length statistics
    report.append("### 3.1 Label Length Distribution\n")
    report.append("| Split | Min | Max | Mean | Median |")
    report.append("|-------|-----|-----|------|--------|")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, split in zip(axes, ['train', 'val', 'test']):
        lengths = label_lengths[split]
        report.append(f"| {split.upper():5s} | {min(lengths):3d} | {max(lengths):3d} | {np.mean(lengths):5.1f} | {np.median(lengths):5.0f} |")
        
        ax.hist(lengths, bins=range(0, int(max(lengths))+2), alpha=0.7, edgecolor='black')
        ax.set_title(f'{split.upper()} Label Lengths')
        ax.set_xlabel('Characters per word')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    
    plt.suptitle('Label Length Distribution by Split', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_lengths.png'), dpi=150)
    plt.close()
    report.append(f"\n![Label lengths](label_lengths.png)\n")
    
    # Character frequency analysis
    report.append("### 3.2 Character Frequency Analysis\n")
    
    char_counts = Counter(all_labels)
    unique_chars = len(char_counts)
    total_chars = len(all_labels)
    
    report.append(f"- **Total characters:** {total_chars:,}")
    report.append(f"- **Unique characters:** {unique_chars}")
    report.append(f"- **Space character (code 1):** {char_counts.get(1, 0):,}")
    
    # Exclude padding (0) and space (1) for character analysis
    char_only = {k: v for k, v in char_counts.items() if k not in [0, 1]}
    report.append(f"- **Actual Amharic characters:** {len(char_only)}")
    
    # Top characters
    top_30 = char_counts.most_common(30)
    
    plt.figure(figsize=(14, 5))
    codes, counts = zip(*top_30)
    plt.bar(range(len(codes)), counts, alpha=0.7, edgecolor='black')
    plt.xticks(range(len(codes)), [str(c) for c in codes], rotation=90, fontsize=8)
    plt.title('Top 30 Most Frequent Character Codes')
    plt.xlabel('Code')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'char_frequencies.png'), dpi=150)
    plt.close()
    report.append(f"\n![Character frequencies](char_frequencies.png)\n")
    
    # Rare characters
    report.append("### 3.3 Rare Character Analysis\n")
    thresholds = [1, 5, 10, 50, 100]
    report.append("| Occurrences ≤ | Characters | % of Total |")
    report.append("|---------------|------------|------------|")
    for thresh in thresholds:
        rare_count = sum(1 for c, cnt in char_only.items() if cnt <= thresh)
        report.append(f"| {thresh:3d} | {rare_count:4d} | {rare_count/len(char_only)*100:5.1f}% |")
    
    # List ultra-rare (1 occurrence)
    ultra_rare = [c for c, cnt in char_only.items() if cnt == 1]
    if ultra_rare:
        report.append(f"\n**Characters appearing only once:** {ultra_rare}")
    
    # Save frequencies
    freq_data = {
        'total_chars': total_chars,
        'unique_chars': unique_chars,
        'space_code': 1,
        'characters_excluding_space': len(char_only),
        'frequencies': {int(k): int(v) for k, v in char_counts.most_common()}
    }
    with open(os.path.join(output_dir, 'char_frequencies.json'), 'w', encoding='utf-8') as f:
        json.dump(freq_data, f, ensure_ascii=False, indent=2)
    
    return char_counts


# -----------------------------------------------------------------------------
# 4. Visual samples
# -----------------------------------------------------------------------------
def visualize_samples(files, output_dir, report, num_samples=5):
    """Create visual montage of samples from each split."""
    report.append("## 4. Visual Samples\n")
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
    
    for row, split in enumerate(['train', 'val', 'test']):
        X = files[f'X_{split}']
        y = files[f'y_{split}']
        
        indices = np.random.choice(X.shape[0], num_samples, replace=False)
        
        for col, idx in enumerate(indices):
            img = X[idx]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            axes[row, col].imshow(img, cmap='gray', interpolation='nearest')
            
            # Label
            seq = y[idx]
            nonzero = seq[seq != 0]
            label = ' '.join(str(c) for c in nonzero[:8])
            axes[row, col].set_title(f'{label}...' if len(nonzero) > 8 else label, fontsize=7)
            axes[row, col].axis('off')
    
    axes[0, 0].set_ylabel('TRAIN', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('VAL', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('TEST', fontsize=12, fontweight='bold')
    
    plt.suptitle('Random Samples from Handwritten Amharic Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_montage.png'), dpi=200)
    plt.close()
    report.append(f"\n![Sample montage](samples_montage.png)\n")
    
    # Also save some individual high-res samples
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(10):
        split = np.random.choice(['train', 'val', 'test'])
        X = files[f'X_{split}']
        y = files[f'y_{split}']
        idx = np.random.randint(0, X.shape[0])
        
        row, col = i // 5, i % 5
        img = X[idx]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        axes[row, col].imshow(img, cmap='gray')
        seq = y[idx]
        nonzero = seq[seq != 0]
        label = ' '.join(str(c) for c in nonzero)
        axes[row, col].set_title(f'{split.upper()} [{len(nonzero)} chars]', fontsize=9)
        axes[row, col].set_xlabel(textwrap.fill(label, width=30), fontsize=7)
        axes[row, col].axis('off')
    
    plt.suptitle('High-Resolution Handwritten Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_highres.png'), dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# 5. Augmentation detection
# -----------------------------------------------------------------------------
def analyze_augmentation(files, output_dir, report):
    """Try to distinguish original vs augmented samples in the training set."""
    report.append("## 5. Augmentation Analysis\n")
    
    # Dataset description: 12,064 original, 21,608 augmented (in 32x128 version)
    # The 64x256 version might have different counts
    
    X_train = files['X_train']
    
    # Simple metrics to detect augmentation
    report.append("### 5.1 Image Sharpness Analysis (Laplacian variance)\n")
    
    from scipy import ndimage
    sample_size = min(2000, X_train.shape[0])
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    
    sharpness_scores = []
    for idx in indices:
        img = X_train[idx]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        sharpness_scores.append(ndimage.laplace(img.astype(np.float32)).var())
    
    sharpness_scores = np.array(sharpness_scores)
    
    plt.figure(figsize=(10, 5))
    plt.hist(sharpness_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Sharpness Distribution (Laplacian Variance) - TRAIN Set')
    plt.xlabel('Sharpness Score')
    plt.ylabel('Frequency')
    plt.axvline(np.median(sharpness_scores), color='red', linestyle='--', 
                label=f'Median: {np.median(sharpness_scores):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sharpness_distribution.png'), dpi=150)
    plt.close()
    report.append(f"\n![Sharpness distribution](sharpness_distribution.png)\n")
    
    report.append(f"- Median sharpness: {np.median(sharpness_scores):.6f}")
    report.append(f"- Sharpness range: [{sharpness_scores.min():.6f}, {sharpness_scores.max():.6f}]")
    report.append(f"- Low sharpness (<0.01) might indicate blur augmentation: {np.sum(sharpness_scores < 0.01)}/{sample_size}")


# -----------------------------------------------------------------------------
# 6. Split consistency check
# -----------------------------------------------------------------------------
def check_split_consistency(files, report):
    """Verify no label leakage between splits."""
    report.append("## 6. Split Consistency Check\n")
    
    # Compare character sets across splits
    char_sets = {}
    for split in ['train', 'val', 'test']:
        y = files[f'y_{split}']
        chars = set()
        for i in range(y.shape[0]):
            chars.update(y[i][y[i] != 0].tolist())
        char_sets[split] = chars
    
    report.append(f"- **Train unique characters:** {len(char_sets['train'])}")
    report.append(f"- **Val unique characters:** {len(char_sets['val'])}")
    report.append(f"- **Test unique characters:** {len(char_sets['test'])}")
    
    # Characters in val/test but not in train
    val_only = char_sets['val'] - char_sets['train']
    test_only = char_sets['test'] - char_sets['train']
    
    if val_only:
        report.append(f"- ⚠️ Characters in VAL but not TRAIN: {sorted(val_only)}")
    else:
        report.append("- ✅ All VAL characters present in TRAIN")
    
    if test_only:
        report.append(f"- ⚠️ Characters in TEST but not TRAIN: {sorted(test_only)}")
    else:
        report.append("- ✅ All TEST characters present in TRAIN")
    
    # Check for exact duplicate labels between splits
    report.append("\n### 6.1 Duplicate Labels Across Splits\n")
    
    import hashlib
    
    def get_hash(seq):
        clean = tuple(c for c in seq if c != 0)
        return hashlib.md5(str(clean).encode()).hexdigest()
    
    train_hashes = set()
    for i in range(files['y_train'].shape[0]):
        train_hashes.add(get_hash(files['y_train'][i]))
    
    val_overlap = 0
    for i in range(files['y_val'].shape[0]):
        if get_hash(files['y_val'][i]) in train_hashes:
            val_overlap += 1
    
    test_overlap = 0
    for i in range(files['y_test'].shape[0]):
        if get_hash(files['y_test'][i]) in train_hashes:
            test_overlap += 1
    
    report.append(f"- VAL labels also in TRAIN: {val_overlap}/{files['y_val'].shape[0]}")
    report.append(f"- TEST labels also in TRAIN: {test_overlap}/{files['y_test'].shape[0]}")


# -----------------------------------------------------------------------------
# 7. Generate report
# -----------------------------------------------------------------------------
def generate_report(output_dir, report_lines):
    """Write the full report to Markdown."""
    report_path = os.path.join(output_dir, 'handwritten_inspection_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Amharic Handwritten Dataset Inspection Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write('\n'.join(report_lines))
    print(f"\n✅ Report saved to {report_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deep inspection of Amharic Handwritten dataset.")
    parser.add_argument('--data_dir', type=str, default='./npy64by256',
                        help='Directory containing the .npy files')
    parser.add_argument('--output_dir', type=str, default='./handwritten_inspection',
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Loading Handwritten Amharic Dataset (64×256)")
    print("=" * 60)
    
    files = load_data(args.data_dir)
    
    report = []
    
    # Run all analyses
    basic_statistics(files, args.output_dir, report)
    pixel_statistics(files, args.output_dir, report)
    char_counts = analyze_labels(files, args.output_dir, report)
    visualize_samples(files, args.output_dir, report)
    analyze_augmentation(files, args.output_dir, report)
    check_split_consistency(files, report)
    
    # Recommendations
    report.append("## 7. Recommendations for Training\n")
    report.append("- **Input normalization:** Scale pixel values to [0,1] based on observed max.")
    report.append("- **Architecture:** CRNN-CTC with input 64×256, CNN downsampling factor ≤8.")
    report.append("- **Character set:** Use all characters from the training set + CTC blank.")
    report.append("- **Augmentation:** Training set already contains augmented samples; additional augmentation may improve generalization.")
    report.append("- **Validation:** Use the provided VAL split; early stopping based on VAL loss.")
    
    generate_report(args.output_dir, report)
    
    print("=" * 60)
    print("Inspection complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
