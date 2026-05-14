#!/usr/bin/env python3
"""
test_model_amharic.py – Evaluate Amharic OCR model with proper Amharic text output.
Uses the mapping you provided to convert integer codes to readable Amharic characters.

Usage:
    python test_model_amharic.py [--model_path PATH] [--data_dir PATH] [--output_dir PATH]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm

# ----------------------------------------------------------------------
# The mapping from your dataset (code → Unicode character)
# ----------------------------------------------------------------------
CODE_TO_CHAR = {
    0: '\n',       # newline / padding separator
    1: ' ',        # space
    2: '!', 3: '(', 4: ')', 5: '-', 6: '/', 7: ':', 8: '?', 9: '\\',
    10: 'o', 11: '«', 12: '²', 13: '·',
    14: 'ሀ', 15: 'ሁ', 16: 'ሂ', 17: 'ሃ', 18: 'ሄ', 19: 'ህ', 20: 'ሆ',
    21: 'ለ', 22: 'ሉ', 23: 'ሊ', 24: 'ላ', 25: 'ሌ', 26: 'ል', 27: 'ሎ', 28: 'ሏ',
    29: 'ሐ', 30: 'ሑ', 31: 'ሒ', 32: 'ሓ', 33: 'ሔ', 34: 'ሕ',
    35: 'መ', 36: 'ሙ', 37: 'ሚ', 38: 'ማ', 39: 'ሜ', 40: 'ም', 41: 'ሞ', 42: 'ሟ',
    43: 'ሠ', 44: 'ሡ', 45: 'ሢ', 46: 'ሣ', 47: 'ሤ', 48: 'ሥ', 49: 'ሦ',
    50: 'ረ', 51: 'ሩ', 52: 'ሪ', 53: 'ራ', 54: 'ሬ', 55: 'ር', 56: 'ሮ', 57: 'ሯ',
    58: 'ሰ', 59: 'ሱ', 60: 'ሲ', 61: 'ሳ', 62: 'ሴ', 63: 'ስ', 64: 'ሶ', 65: 'ሷ',
    66: 'ሸ', 67: 'ሹ', 68: 'ሺ', 69: 'ሻ', 70: 'ሼ', 71: 'ሽ', 72: 'ሾ', 73: 'ሿ',
    74: 'ቀ', 75: 'ቁ', 76: 'ቂ', 77: 'ቃ', 78: 'ቄ', 79: 'ቅ', 80: 'ቆ', 81: 'ቋ',
    82: 'ቐ',
    83: 'በ', 84: 'ቡ', 85: 'ቢ', 86: 'ባ', 87: 'ቤ', 88: 'ብ', 89: 'ቦ', 90: 'ቧ',
    91: 'ቨ', 92: 'ቪ', 93: 'ቫ', 94: 'ቬ', 95: 'ቭ', 96: 'ቮ',
    97: 'ተ', 98: 'ቱ', 99: 'ቲ', 100: 'ታ', 101: 'ቴ', 102: 'ት', 103: 'ቶ', 104: 'ቷ',
    105: 'ቸ', 106: 'ቹ', 107: 'ቺ', 108: 'ቻ', 109: 'ቼ', 110: 'ች', 111: 'ቾ', 112: 'ቿ',
    113: 'ኀ', 114: 'ኃ', 115: 'ኅ', 116: 'ኋ',
    117: 'ነ', 118: 'ኑ', 119: 'ኒ', 120: 'ና', 121: 'ኔ', 122: 'ን', 123: 'ኖ', 124: 'ኗ',
    125: 'ኘ', 126: 'ኙ', 127: 'ኚ', 128: 'ኛ', 129: 'ኜ', 130: 'ኝ', 131: 'ኞ', 132: 'ኟ',
    133: 'አ', 134: 'ኡ', 135: 'ኢ', 136: 'ኣ', 137: 'ኤ', 138: 'እ', 139: 'ኦ', 140: 'ኧ',
    141: 'ከ', 142: 'ኩ', 143: 'ኪ', 144: 'ካ', 145: 'ኬ', 146: 'ክ', 147: 'ኮ', 148: 'ኰ',
    149: 'ኲ', 150: 'ኳ',
    151: 'ኸ', 152: 'ኻ', 153: 'ኼ', 154: 'ኽ',
    155: 'ወ', 156: 'ዉ', 157: 'ዊ', 158: 'ዋ', 159: 'ዌ', 160: 'ው', 161: 'ዎ',
    162: 'ዐ', 163: 'ዑ', 164: 'ዒ', 165: 'ዓ', 166: 'ዔ', 167: 'ዕ', 168: 'ዖ',
    169: 'ዘ', 170: 'ዙ', 171: 'ዚ', 172: 'ዛ', 173: 'ዜ', 174: 'ዝ', 175: 'ዞ', 176: 'ዟ',
    177: 'ዠ', 178: 'ዢ', 179: 'ዣ', 180: 'ዤ', 181: 'ዥ', 182: 'ዦ', 183: 'ዧ',
    184: 'የ', 185: 'ዩ', 186: 'ዪ', 187: 'ያ', 188: 'ዬ', 189: 'ይ', 190: 'ዮ',
    191: 'ደ', 192: 'ዱ', 193: 'ዲ', 194: 'ዳ', 195: 'ዴ', 196: 'ድ', 197: 'ዶ', 198: 'ዷ',
    199: 'ዼ', 200: 'ዾ',
    201: 'ጀ', 202: 'ጁ', 203: 'ጂ', 204: 'ጃ', 205: 'ጄ', 206: 'ጅ', 207: 'ጆ', 208: 'ጇ',
    209: 'ገ', 210: 'ጉ', 211: 'ጊ', 212: 'ጋ', 213: 'ጌ', 214: 'ግ', 215: 'ጎ', 216: 'ጐ',
    217: 'ጓ',
    218: 'ጠ', 219: 'ጡ', 220: 'ጢ', 221: 'ጣ', 222: 'ጤ', 223: 'ጥ', 224: 'ጦ', 225: 'ጧ',
    226: 'ጨ', 227: 'ጩ', 228: 'ጪ', 229: 'ጫ', 230: 'ጬ', 231: 'ጭ', 232: 'ጮ', 233: 'ጯ',
    234: 'ጱ', 235: 'ጲ', 236: 'ጳ', 237: 'ጴ', 238: 'ጵ',
    239: 'ጸ', 240: 'ጹ', 241: 'ጻ', 242: 'ጼ', 243: 'ጽ', 244: 'ጾ', 245: 'ጿ',
    246: 'ፀ', 247: 'ፁ', 248: 'ፂ', 249: 'ፃ', 250: 'ፄ', 251: 'ፅ', 252: 'ፆ',
    253: 'ፈ', 254: 'ፉ', 255: 'ፊ', 256: 'ፋ', 257: 'ፌ', 258: 'ፍ', 259: 'ፎ', 260: 'ፏ',
    261: 'ፐ', 262: 'ፑ', 263: 'ፒ', 264: 'ፓ', 265: 'ፔ', 266: 'ፕ', 267: 'ፖ',
    268: '፡',    # Amharic word separator (፡)
    269: '።', 270: '፣', 271: '፤', 272: '፥', 273: '፦',
    274: '–', 275: '—', 276: '\u2018', 277: '\u2019', 278: '›', 279: 'ⶰ',
}


# ----------------------------------------------------------------------
# Model definition (unchanged)
# ----------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.cnn_out_height = img_height // 16
        self.rnn = nn.LSTM(512 * self.cnn_out_height, hidden_size, 2,
                           bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        return nn.functional.log_softmax(self.classifier(rnn_out), dim=2)


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def preprocess_image(img, target_height=64, target_width=256, normalize_max=27.1667):
    if img.shape == (128, 48):
        img = img.T
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / normalize_max
    img = np.clip(img, 0.0, 1.0)
    img = np.expand_dims(img, axis=0)
    return img


# ----------------------------------------------------------------------
# Character mapping
# ----------------------------------------------------------------------
def build_char_mappings(code_counts):
    sorted_codes = [code for code, _ in code_counts.most_common()]
    if 1 not in sorted_codes:
        sorted_codes.insert(0, 1)
    char_to_idx = {code: i for i, code in enumerate(sorted_codes)}
    idx_to_char = {i: code for code, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def greedy_decode(log_probs, blank_idx):
    pred = log_probs.argmax(dim=1).cpu().numpy()
    decoded = []
    prev = -1
    for p in pred:
        if p != blank_idx and p != prev:
            decoded.append(p)
        prev = p
    return decoded


def codes_to_amharic(codes):
    """Convert list of integer codes to Amharic string."""
    return ''.join(CODE_TO_CHAR.get(int(c), f'<{c}>') for c in codes)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(
                curr_row[-1] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + (c1 != c2)
            ))
        prev_row = curr_row
    return prev_row[-1]


# ----------------------------------------------------------------------
# Highlight differences between two Amharic strings
# ----------------------------------------------------------------------
def highlight_errors(target_codes, pred_codes):
    """
    Return two strings with ANSI color highlighting:
    - Green for correct characters
    - Red for substitutions
    - Yellow for insertions/deletions
    """
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    # Simple character-by-character alignment
    result_true = []
    result_pred = []
    
    max_len = max(len(target_codes), len(pred_codes))
    
    for i in range(max_len):
        t = target_codes[i] if i < len(target_codes) else None
        p = pred_codes[i] if i < len(pred_codes) else None
        
        t_char = CODE_TO_CHAR.get(t, '?') if t is not None else '·'
        p_char = CODE_TO_CHAR.get(p, '?') if p is not None else '·'
        
        if t == p:
            result_true.append(f"{GREEN}{t_char}{RESET}")
            result_pred.append(f"{GREEN}{p_char}{RESET}")
        elif t is None:
            result_true.append(f"{YELLOW}·{RESET}")
            result_pred.append(f"{YELLOW}{p_char}{RESET}")
        elif p is None:
            result_true.append(f"{YELLOW}{t_char}{RESET}")
            result_pred.append(f"{YELLOW}·{RESET}")
        else:
            result_true.append(f"{RED}{t_char}{RESET}")
            result_pred.append(f"{RED}{p_char}{RESET}")
    
    return ''.join(result_true), ''.join(result_pred)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Amharic OCR with readable Amharic output")
    parser.add_argument("--model_path", type=str,
                        default="/home/lo/Desktop/vscode/C++_NLP/amharicCV/output/20260426_232133/best_model.pth")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.expanduser("~/Desktop/vscode/C++_NLP/amharicCV/train_big/train"))
    parser.add_argument("--output_dir", type=str, default="./test_evaluation_amharic")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show_errors", action="store_true", default=True,
                        help="Show color‑highlighted character errors")
    parser.add_argument("--num_examples", type=int, default=20,
                        help="Number of examples to show in detail")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X_path = os.path.join(args.data_dir, "X_trainp_pg_vg.npy")
    y_path = os.path.join(args.data_dir, "y_trainp_pg_vg.npy")
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: data files not found in {args.data_dir}")
        sys.exit(1)

    print("Loading data...")
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')

    # Build character mapping
    all_codes = y[y != 0]
    code_counts = Counter(all_codes)
    char_to_idx, idx_to_char = build_char_mappings(code_counts)
    num_classes = len(char_to_idx) + 1
    blank_idx = num_classes - 1

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = CRNN(num_classes=num_classes, img_height=64, hidden_size=256).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare test set
    total_printed = 40929
    np.random.seed(args.seed)
    indices = np.random.permutation(total_printed)
    n_test = int(total_printed * args.test_split)
    test_indices = indices[-n_test:]
    print(f"Test samples: {len(test_indices)}")

    total_chars = 0
    total_errors = 0
    all_cer = []
    perfect_count = 0
    examples = []

    print("Evaluating...")
    for idx in tqdm(test_indices, desc="Test"):
        # Ground truth
        label_seq = y[idx]
        target_codes = label_seq[label_seq != 0].astype(np.int64).tolist()
        target_idx = [char_to_idx[c] for c in target_codes if c in char_to_idx]

        # Preprocess and infer
        img_raw = X[idx]
        img_proc = preprocess_image(img_raw)
        img_tensor = torch.from_numpy(img_proc).unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(img_tensor).squeeze(0)

        pred_idx = greedy_decode(log_probs, blank_idx)
        pred_codes = [idx_to_char[i] for i in pred_idx if i in idx_to_char]

        dist = levenshtein_distance(pred_codes, target_codes)
        cer = dist / len(target_codes) if len(target_codes) > 0 else 0.0
        
        total_errors += dist
        total_chars += len(target_codes)
        all_cer.append(cer)
        
        if cer == 0:
            perfect_count += 1

        if len(examples) < args.num_examples:
            true_text = codes_to_amharic(target_codes)
            pred_text = codes_to_amharic(pred_codes)
            examples.append({
                'index': int(idx),
                'true_text': true_text,
                'pred_text': pred_text,
                'target_codes': target_codes,
                'pred_codes': pred_codes,
                'cer': cer,
                'correct': cer == 0
            })

    overall_cer = total_errors / total_chars if total_chars > 0 else 0.0
    avg_cer = np.mean(all_cer) if all_cer else 0.0

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                    AMHARIC OCR EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Test samples       : {len(all_cer):,}")
    print(f"  Perfect predictions: {perfect_count:,} ({perfect_count/len(all_cer)*100:.1f}%)")
    print(f"  Average CER        : {avg_cer:.4f}")
    print(f"  Overall CER        : {overall_cer:.4f}")
    print(f"  Total errors       : {total_errors:,} / {total_chars:,} characters")
    print("=" * 70)

    # Show examples
    print(f"\n{'─' * 70}")
    print(f"  AMHARIC TEXT PREDICTIONS (first {min(args.num_examples, 15)} examples)")
    print(f"{'─' * 70}")
    
    for i, ex in enumerate(examples[:15]):
        status = "✅" if ex['correct'] else f"⚠️ CER={ex['cer']:.3f}"
        print(f"\n  Sample {ex['index']}  {status}")
        print(f"  {'Truth:':8s} {ex['true_text']}")
        print(f"  {'Pred:':8s} {ex['pred_text']}")
        
        if not ex['correct'] and args.show_errors:
            true_hl, pred_hl = highlight_errors(ex['target_codes'], ex['pred_codes'])
            print(f"  {'Diff:':8s} {true_hl}")
            print(f"  {'':8s} {pred_hl}")

    # Save detailed results
    results_path = os.path.join(args.output_dir, "test_results_amharic.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("AMHARIC OCR EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test samples: {len(all_cer):,}\n")
        f.write(f"Perfect predictions: {perfect_count:,} ({perfect_count/len(all_cer)*100:.1f}%)\n")
        f.write(f"Average CER: {avg_cer:.4f}\n")
        f.write(f"Overall CER: {overall_cer:.4f}\n")
        f.write(f"Total errors: {total_errors:,} / {total_chars:,} characters\n\n")
        f.write("─" * 70 + "\n")
        f.write("ALL PREDICTIONS\n")
        f.write("─" * 70 + "\n\n")
        for ex in examples:
            status = "PERFECT" if ex['correct'] else f"CER={ex['cer']:.4f}"
            f.write(f"Sample {ex['index']}  [{status}]\n")
            f.write(f"  Truth: {ex['true_text']}\n")
            f.write(f"  Pred : {ex['pred_text']}\n\n")

    print(f"\n📁 Full results saved to: {results_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
