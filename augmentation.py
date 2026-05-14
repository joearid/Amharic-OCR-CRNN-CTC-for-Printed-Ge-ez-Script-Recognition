"""Data augmentation pipeline using albumentations."""
import numpy as np
from typing import Optional, Dict, Any
from .config import OCRConfig

def get_augmentation_pipeline(config: OCRConfig) -> Optional[Any]:
    """Return augmentation pipeline, or None if disabled."""
    if not config.use_augmentation:
        return None
    try:
        import albumentations as A
    except ImportError:
        print("WARNING: albumentations not installed. Augmentation disabled.")
        return None
    return A.Compose([
        A.Rotate(limit=config.rotation_degrees, border_mode=0, p=0.5),
        A.ElasticTransform(alpha=config.elastic_alpha, sigma=config.elastic_sigma, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.contrast_limit, p=0.5),
    ])
