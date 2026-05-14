# amharic_ocr/__init__.py
"""
Amharic OCR Package using CRNN-CTC architecture.
"""

# Make all modules available at package level
from .config import OCRConfig
from .dataset import AmharicOCRDataset, create_datasets, build_char_mappings
from .utils import analyze_dataset, deduplicate_dataset
from .preprocessing import preprocess_image, decode_label_sequence
from .augmentation import get_augmentation_pipeline
from .train import CRNN, CTCTrainer, collate_ctc

__version__ = "0.1.0"
__all__ = [
    "OCRConfig",
    "AmharicOCRDataset", 
    "create_datasets",
    "build_char_mappings",
    "analyze_dataset",
    "deduplicate_dataset",
    "preprocess_image",
    "decode_label_sequence",
    "get_augmentation_pipeline",
    "CRNN",
    "CTCTrainer",
    "collate_ctc"
]
