import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OCRConfig:
    # Paths
    data_dir: str = os.path.expanduser("~/Desktop/vscode/C++_NLP/amharicCV/train_big/train")
    output_dir: str = "./amharic_ocr_output"
    mapping_file: Optional[str] = None   # JSON mapping codes -> Unicode

    # Image processing
    img_height: int = 64
    img_width: int = 256
    normalize_max: float = 27.1667       # From inspection

    # Dataset
    val_split: float = 0.1               # Use only real printed data for validation
    rare_threshold: int = 100            # Characters with fewer occurrences are "rare"
    rare_weight_multiplier: float = 5.0  # Weight multiplier for lines containing rare chars

    # Augmentation
    use_augmentation: bool = False  # Disable for now    rotation_degrees: float = 2.0
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1

    # Model
    num_classes: int = 280               # 278 chars + space + CTC blank
    cnn_backbone: str = "vgg"            # Options: "vgg", "resnet"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 15
    device: str = "cuda"

    # Post‑processing
    beam_width: int = 10
    use_lm: bool = True                  # Use n‑gram language model in beam search
    ngram_file: str = "ngrams.json"      # From phase2 inspection
