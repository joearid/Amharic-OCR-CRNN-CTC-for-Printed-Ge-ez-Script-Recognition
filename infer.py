#!/usr/bin/env python3
"""
infer_image2.py – Enhanced Amharic OCR Inference (FIXED)
Restores working preprocessing + adds safe enhancements & cleanup.
"""
import argparse, json, os, sys, re
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import cv2

# ── Complete CODE_TO_CHAR (280 classes) ───────────────────────────────────
CODE_TO_CHAR = {
    0: '\n', 1: ' ', 2: '!', 3: '(', 4: ')', 5: '-', 6: '/', 7: ':', 8: '?', 9: '\\',
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
    82: 'ቐ', 83: 'በ', 84: 'ቡ', 85: 'ቢ', 86: 'ባ', 87: 'ቤ', 88: 'ብ', 89: 'ቦ', 90: 'ቧ',
    91: 'ቨ', 92: 'ቪ', 93: 'ቫ', 94: 'ቬ', 95: 'ቭ', 96: 'ቮ',
    97: 'ተ', 98: 'ቱ', 99: 'ቲ', 100: 'ታ', 101: 'ቴ', 102: 'ት', 103: 'ቶ', 104: 'ቷ',
    105: 'ቸ', 106: 'ቹ', 107: 'ቺ', 108: 'ቻ', 109: 'ቼ', 110: 'ች', 111: 'ቾ', 112: 'ቿ',
    113: 'ኀ', 114: 'ኃ', 115: 'ኅ', 116: 'ኋ',
    117: 'ነ', 118: 'ኑ', 119: 'ኒ', 120: 'ና', 121: 'ኔ', 122: 'ን', 123: 'ኖ', 124: 'ኗ',
    125: 'ኘ', 126: 'ኙ', 127: 'ኚ', 128: 'ኛ', 129: 'ኜ', 130: 'ኝ', 131: 'ኞ', 132: 'ኟ',
    133: 'አ', 134: 'ኡ', 135: 'ኢ', 136: 'ኣ', 137: 'ኤ', 138: 'እ', 139: 'ኦ', 140: 'ኧ',
    141: 'ከ', 142: 'ኩ', 143: 'ኪ', 144: 'ካ', 145: 'ኬ', 146: 'ክ', 147: 'ኮ', 148: 'ኰ',
    149: 'ኲ', 150: 'ኳ', 151: 'ኸ', 152: 'ኻ', 153: 'ኼ', 154: 'ኽ',
    155: 'ወ', 156: 'ዉ', 157: 'ዊ', 158: 'ዋ', 159: 'ዌ', 160: 'ው', 161: 'ዎ',
    162: 'ዐ', 163: 'ዑ', 164: 'ዒ', 165: 'ዓ', 166: 'ዔ', 167: 'ዕ', 168: 'ዖ',
    169: 'ዘ', 170: 'ዙ', 171: 'ዚ', 172: 'ዛ', 173: 'ዜ', 174: 'ዝ', 175: 'ዞ', 176: 'ዟ',
    177: 'ዠ', 178: 'ዢ', 179: 'ዣ', 180: 'ዤ', 181: 'ዥ', 182: 'ዦ', 183: 'ዧ',
    184: 'የ', 185: 'ዩ', 186: 'ዪ', 187: 'ያ', 188: 'ዬ', 189: 'ይ', 190: 'ዮ',
    191: 'ደ', 192: 'ዱ', 193: 'ዲ', 194: 'ዳ', 195: 'ዴ', 196: 'ድ', 197: 'ዶ', 198: 'ዷ',
    199: 'ዼ', 200: 'ዾ',
    201: 'ጀ', 202: 'ጁ', 203: 'ጂ', 204: 'ጃ', 205: 'ጄ', 206: 'ጅ', 207: 'ጆ', 208: 'ጇ',
    209: 'ገ', 210: 'ጉ', 211: 'ጊ', 212: 'ጋ', 213: 'ጌ', 214: 'ግ', 215: 'ጎ', 216: 'ጐ', 217: 'ጓ',
    218: 'ጠ', 219: 'ጡ', 220: 'ጢ', 221: 'ጣ', 222: 'ጤ', 223: 'ጥ', 224: 'ጦ', 225: 'ጧ',
    226: 'ጨ', 227: 'ጩ', 228: 'ጪ', 229: 'ጫ', 230: 'ጬ', 231: 'ጭ', 232: 'ጮ', 233: 'ጯ',
    234: 'ጱ', 235: 'ጲ', 236: 'ጳ', 237: 'ጴ', 238: 'ጵ',
    239: 'ጸ', 240: 'ጹ', 241: 'ጻ', 242: 'ጼ', 243: 'ጽ', 244: 'ጾ', 245: 'ጿ',
    246: 'ፀ', 247: 'ፁ', 248: 'ፂ', 249: 'ፃ', 250: 'ፄ', 251: 'ፅ', 252: 'ፆ',
    253: 'ፈ', 254: 'ፉ', 255: 'ፊ', 256: 'ፋ', 257: 'ፌ', 258: 'ፍ', 259: 'ፎ', 260: 'ፏ',
    261: 'ፐ', 262: 'ፑ', 263: 'ፒ', 264: 'ፓ', 265: 'ፔ', 266: 'ፕ', 267: 'ፖ',
    268: '፡', 269: '።', 270: '፣', 271: '፤', 272: '፥', 273: '፦',
    274: '–', 275: '—', 276: '\u2018', 277: '\u2019', 278: '›', 279: 'ⶰ',
}

# ── Training Constants (DO NOT CHANGE) ─────────────────────────────────────
TRAIN_H = 64
TRAIN_W = 256
TRAIN_BG_VAL = 2.7824
NORMALIZE_MAX = 27.166653885613126

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
        )
        self.cnn_out_height = img_height // 16
        self.rnn = nn.LSTM(512 * self.cnn_out_height, hidden_size, 2,
                           bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c*h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        return nn.functional.log_softmax(self.classifier(rnn_out), dim=2)

def enhance_and_preprocess_line(pil_img, use_clahe=True):
    """
    EXACT working preprocessing + optional safe enhancement.
    1. Preserves aspect ratio (height=64, width scales proportionally)
    2. Scales to exact training distribution [0, 0.1024]
    3. Optional CLAHE for contrast (clamped to 0-255 before scaling)
    """
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    
    # Optional: Light CLAHE to boost local contrast without breaking distribution
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        arr = clahe.apply(arr.astype(np.uint8)).astype(np.float32)
    
    # KEY FIX: Preserve aspect ratio (don't squash 670px → 256px)
    h, w = arr.shape
    target_h = TRAIN_H
    target_w = max(1, int(w * (target_h / h)))
    arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # EXACT training distribution match
    arr = (arr / 255.0) * TRAIN_BG_VAL   # → 0..2.7824
    arr = arr / NORMALIZE_MAX             # → 0..0.1024
    arr = np.clip(arr, 0.0, 1.0)
    
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, W]

def segment_lines(image_path):
    img_pil = Image.open(image_path).convert("L")
    arr = np.array(img_pil, dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    dilated = cv2.dilate(binary, h_kernel, iterations=1)
    row_sums = dilated.sum(axis=1)
    in_line, start, lines = False, 0, []
    for i, s in enumerate(row_sums):
        if s > 0 and not in_line:
            in_line, start = True, i
        elif s == 0 and in_line:
            in_line = False
            lines.append((start, i))
    if in_line:
        lines.append((start, len(row_sums)))
    if not lines:
        return [img_pil]
    merged = [lines[0]]
    for y0, y1 in lines[1:]:
        if y0 - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], y1)
        else:
            merged.append((y0, y1))
    merged = [(y0, y1) for y0, y1 in merged if y1 - y0 >= 8]
    pad = 4
    crops = []
    for y0, y1 in merged:
        y0p = max(0, y0 - pad)
        y1p = min(arr.shape[0], y1 + pad)
        crops.append(img_pil.crop((0, y0p, img_pil.width, y1p)))
    os.makedirs("debug_lines", exist_ok=True)
    for i, c in enumerate(crops):
        c.save(f"debug_lines/line_{i+1:03d}.png")
    print(f"  [debug] {len(crops)} line crops → ./debug_lines/")
    return crops

def ctc_greedy_decode(log_probs, idx_to_char, blank_idx):
    indices = log_probs.argmax(dim=2).squeeze(0).tolist()
    decoded, prev = [], -1
    for idx in indices:
        if idx != prev and idx != blank_idx:
            code = idx_to_char.get(idx)
            if code is not None:
                ch = CODE_TO_CHAR.get(code, f'[{code}]')
                if ch != '\n':
                    decoded.append(ch)
        prev = idx
    return ''.join(decoded)

def clean_amharic_text(text: str) -> str:
    """Fixes CTC artifacts and Amharic punctuation spacing."""
    if not text: return text
    text = re.sub(r'(.)\1+', r'\1', text)
    confusions = {'ገአ': 'የ', 'አየ': 'ኢየ', 'ዘዘ': 'የ', 'ሀ': 'ህ', 'ሥ': 'ስ', 'ጊ': 'ጊ'}
    for wrong, right in confusions.items():
        text = text.replace(wrong, right)
    text = re.sub(r'\s+([።፣፤፥፡])', r'\1', text)
    text = re.sub(r'([።፡])\s+', r'\1 ', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", default="metadata.json")
    parser.add_argument("--no_segment", action="store_true")
    parser.add_argument("--no_enhance", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.meta, encoding="utf-8") as f:
        metadata = json.load(f)
    idx_to_char = {int(k): int(v) for k, v in metadata["idx_to_char"].items()}
    num_classes = metadata["num_classes"]
    blank_idx = num_classes - 1

    ckpt = torch.load(args.model, map_location=device)
    num_classes = ckpt["classifier.weight"].shape[0]
    blank_idx = num_classes - 1

    model = CRNN(num_classes=num_classes, img_height=TRAIN_H,
                 hidden_size=args.hidden_size).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded ✅  (input: {TRAIN_H}×VariableWidth)")

    lines = [Image.open(args.image).convert("L")] if args.no_segment else segment_lines(args.image)
    print(f"Lines: {len(lines)}")
    print("=" * 60)

    predictions = []
    use_clahe = not args.no_enhance
    
    with torch.no_grad():
        for i, crop in enumerate(lines):
            tensor = enhance_and_preprocess_line(crop, use_clahe=use_clahe).to(device)
            # Quick sanity check
            if i == 0:
                print(f"  [tensor] min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
            
            log_probs = model(tensor)
            raw = ctc_greedy_decode(log_probs, idx_to_char, blank_idx)
            cleaned = clean_amharic_text(raw)
            predictions.append(cleaned)
            print(f"Line {i+1:>3}: {cleaned if cleaned else '(empty)'}")

    print("=" * 60)
    full_text = '\n'.join(predictions)
    print("\n📄 Complete Text:\n")
    print(full_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"\n💾 Saved to: {args.output}")

if __name__ == "__main__":
    main()
