# Amharic OCR: CRNN-CTC for Printed Ge'ez Script Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete OCR pipeline for printed Amharic text using a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss. Achieves **96.3% line-level accuracy** and **0.64% Character Error Rate** on held-out real printed data.

**Key Innovation:** LLM-augmented OCR pipeline that combines our specialized CRNN with large language model post-processing, outperforming general-purpose vision models on Amharic text.

<p align="center">
  <img src="docs/figures/sample_prediction.png" alt="Sample OCR Result" width="700"/>
</p>

---

## 🎯 Problem

Amharic, written in the Ge'ez script (ግዕዝ), is a liturgical and literary language used by the Ethiopian Orthodox Church and millions of speakers. Despite its cultural importance, **no mainstream OCR solution exists** for Amharic comparable to English OCR systems. The script presents unique challenges:

- **280 unique glyphs** (abugida system: consonant + vowel combinations)
- **Complex morphology** with subtle diacritic distinctions (e.g., ሰ vs ሠ, ሰ vs ስ)
- **Severe class imbalance** (Zipf's law: 40 characters appear ≤10 times)
- **Limited digitized training data** compared to Latin scripts

---

## 🏗️ Architecture

We implement a **CRNN-CTC** architecture inspired by Shi et al. (2017):

```
Input (1×64×256)
    ↓
CNN Backbone (7 conv layers, VGG-style)
    ↓
Feature Map (512×4×64) → Reshape → Sequence (64×2048)
    ↓
2-Layer Bidirectional LSTM (hidden=256) → (64×512)
    ↓
Linear Classifier → (64×280 classes)
    ↓
CTC Loss (alignment-free training)
```

| Component | Details |
|-----------|---------|
| **CNN** | 7 convolutional layers with BatchNorm, ReLU, progressive max-pooling. Output stride: 4× in width, 16× in height |
| **RNN** | 2-layer BiLSTM, 256 hidden units, bidirectional |
| **Output** | 280 classes (278 characters + space + CTC blank) |
| **Parameters** | ~7.5M total |
| **Loss** | CTC Loss (Graves et al., 2006) — no character-level annotations needed |

### Design Choices

| Choice | Justification |
|--------|---------------|
| **CNN backbone** | Learns visual features hierarchically; translation equivariance preserves spatial structure |
| **BiLSTM** | Models sequential context; bidirectional crucial for disambiguating visually similar characters |
| **CTC Loss** | Eliminates need for pre-segmented character bounding boxes; handles variable-length sequences |
| **Weighted Sampling** | Counteracts Zipfian long-tail distribution; rare characters sampled 5-10× more often |

---

## 📊 Dataset

We use the **ADOCR** printed Amharic dataset with additional synthetic data:

| Subset | Samples | Description |
|--------|---------|-------------|
| Printed (Real) | 40,929 | Power Geez font, real documents |
| Synthetic Power Geez | 197,484 | Font-rendered variants |
| Synthetic Visual Geez | 80,293 | Font-rendered variants |
| **Total** | **318,706** | |

- **Vocabulary:** 279 unique codes (278 characters + space)
- **Sequence lengths:** 3–32 characters, mean 17.1
- **Total characters:** 5.45M instances
- **Preprocessing:** Transpose (128×48 → 48×128), resize to 256×64, normalize by 27.1667

<p align="center">
  <img src="docs/figures/class_freq_distribution.png" alt="Zipf Distribution" width="400"/>
  <img src="docs/figures/pixel_distribution_by_subset.png" alt="Pixel Distribution" width="400"/>
</p>

### Class Imbalance & Mitigation

Character frequencies follow **Zipf's Law** — a few characters dominate, while many are extremely rare:

- Max/median frequency ratio: **90:1**
- 40 characters (14.4%) appear **≤10 times**
- 92 characters (33.1%) appear **≤500 times**

**Our solution:** Weighted Random Sampler that oversamples lines containing rare characters by 5-10×.

---

## 📈 Results

### Test Set Evaluation (8,185 real printed samples)

| Metric | Value |
|--------|-------|
| **Perfect Predictions** | 7,883 (96.3%) |
| **Average CER** | 0.0076 |
| **Overall CER** | 0.0064 |
| **Total Errors** | 1,011 / 157,154 characters |

### Sample Predictions

| Status | Truth | Prediction |
|--------|-------|------------|
| ✅ | ኢንዱስትሪ ደካማነት አቶ ማርቆስ | ኢንዱስትሪ ደካማነት አቶ ማርቆስ |
| ✅ | ባህርዳር ቴክኖሎጂ ኢንስቲትዩት | ባህርዳር ቴክኖሎጂ ኢንስቲትዩት |
| ⚠️ CER=0.045 | የሚ**ሰ**ጠው የውሎ አበል... | የሚ**ስ**ጠው የውሎ አበል... |

**Common error:** ሰ (sä) ↔ ስ (sə) — subtle diacritic distinction (~6-pixel difference at 64×256 resolution).

<p align="center">
  <img src="docs/figures/confusion_example.png" alt="Error Analysis" width="600"/>
</p>

---

## 🎬 Demo: Amharic Bible OCR

We test on a real-world document — a page from an Amharic Bible (622×670 px, 20 text lines):

### Pipeline
1. **Image Inspection** — OCR Readiness Score: 85/100
2. **Line Segmentation** — 20 lines detected via projection profile
3. **CRNN Inference** — Raw Amharic text extracted
4. **LLM Correction** — ChatGPT fixes spelling, grammar, theological terminology
5. **Comparison** — Our CRNN+LLM vs. ChatGPT's native OCR

### Results

| System | Quality Score |
|--------|--------------|
| Our CRNN (raw) | ~6,000 / 10,000 |
| ChatGPT OCR (direct) | 5,210 / 10,000 |
| **Our CRNN + LLM Correction** | **9,380 / 10,000** |

> **Our specialized CRNN combined with LLM post-processing significantly outperforms ChatGPT's general-purpose vision model on Amharic text.**

<p align="center">
  <img src="demo/figures/comparison.png" alt="System Comparison" width="700"/>
</p>

---
## To Train:
python amharic_ocr/train.py --epochs 50 --batch_size 16
## The trained model will be saved to output/YYYYMMDD_HHMMSS/best_model.pth.
Expected results after training:
- Character Error Rate (CER): ~0.64%
- Line-level accuracy: ~96%

Evaluate on Test Set

After training, evaluate the model on the held-out test split:
bash

python scripts/evaluate.py \
  --model_path output/YYYYMMDD_HHMMSS/best_model.pth \
  --data_dir ~/Desktop/vscode/C++_NLP/amharicCV/train_big/train

Output:
text

Test samples: 8,185
Perfect predictions: ~96.3%
Average CER: ~0.0076
Overall CER: ~0.0064

Run Inference on Any Image

Use the trained model to extract Amharic text from a document image:
bash

python scripts/infer.py \
  --image path/to/your_document.jpg \
  --model output/YYYYMMDD_HHMMSS/best_model.pth \
  --meta data/metadata.json \
  --output result.txt

Options:
Flag	Description
--no_segment	Skip line detection (use for single-line images)
--no_enhance	Disable CLAHE contrast enhancement
--output	Save extracted text to file

Example output:
text

Lines: 20
============================================================
Line   1: በሙላት ያልተገለጠው መንግሥት
Line   2: በሦስተኛ ደረጃ፣ የእግዚአብሔር መንግሥት...
...

Analyze an Image Before OCR

Inspect image quality, detect skew, find text lines:
bash

python scripts/inspect_image.py path/to/image.jpg --output_dir ./inspection

Generates a report with:

    Pixel statistics and histogram

    OCR readiness score (0-100)

    Line detection visualization

    Skew angle measurement

    Model compatibility check

Generate Character Mapping

If you need to regenerate metadata.json from the dataset:
bash

python scripts/extract_mapping.py

📁 Data Setup

The training script expects the dataset at:
text

~/Desktop/vscode/C++_NLP/amharicCV/train_big/train/
├── X_trainp_pg_vg.npy
└── y_trainp_pg_vg.npy

Update config.py if your data is stored elsewhere:
python

@dataclass
class OCRConfig:
    data_dir: str = "path/to/your/data"
    
## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/amharic-ocr.git
cd amharic-ocr
pip install -r requirements.txt
