import cv2
import numpy as np

def preprocess_image(img: np.ndarray, target_height: int, target_width: int,
                     normalize_max: float, augment_fn=None) -> np.ndarray:
    """
    Preprocess a single image to a fixed size (1, target_height, target_width).
    No augmentation is applied inside here anymore; we'll handle it separately.
    """
    # 1. Transpose (dataset is 128x48 -> 48x128)
    if img.shape == (128, 48):
        img = img.T

    # 2. Resize exactly to target dimensions (force exact size)
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 3. Normalize to [0,1] as float32
    img = img.astype(np.float32) / normalize_max
    img = np.clip(img, 0.0, 1.0)

    # 4. Add channel dimension (1, H, W)
    img = np.expand_dims(img, axis=0)

    return img

def decode_label_sequence(seq: np.ndarray, char_to_idx: dict) -> np.ndarray:
    seq = seq[seq != 0]
    return np.array([char_to_idx[code] for code in seq], dtype=np.int32)
