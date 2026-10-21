"""Single-image I/O, including Unicode Windows paths."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input must be a single image file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    encoded = np.fromfile(path, dtype=np.uint8)
    if not encoded.size:
        raise ValueError(f"Empty image file: {path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Cannot decode image: {path}") from error
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported output image extension: {path.suffix}")
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Cannot encode output image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
