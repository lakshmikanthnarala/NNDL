"""Image preprocessing for SignSpeak's 28 × 28 grayscale CNN input."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


INPUT_SIZE = 28


def _largest_foreground(gray: np.ndarray) -> np.ndarray:
    thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        thresholded, connectivity=8
    )
    if count <= 1:
        return gray
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component_index]
    image_area = gray.shape[0] * gray.shape[1]
    if area < image_area * 0.02 or area > image_area * 0.92:
        return gray
    pad = max(2, int(max(width, height) * 0.08))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + width + pad), min(gray.shape[0], y + height + pad)
    return gray[y0:y1, x0:x1]


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return one normalized model input with shape (1, 28, 28, 1)."""
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 3:
            if array.shape[2] == 1:
                gray = array[:, :, 0]
            else:
                if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                    raise ValueError("Image pixels must be finite numeric values.")
                if array.max(initial=0) <= 1:
                    array = array * 255
                array = np.clip(array, 0, 255).astype(np.uint8)
                if array.shape[2] == 4:
                    array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        elif array.ndim == 2:
            gray = array
        else:
            raise ValueError("Image must be grayscale or RGB.")
        if gray.dtype != np.uint8:
            if not np.issubdtype(gray.dtype, np.number) or not np.isfinite(gray).all():
                raise ValueError("Image pixels must be finite numeric values.")
            if gray.max(initial=0) <= 1:
                gray = gray * 255
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    if gray.size == 0 or min(gray.shape) < 2:
        raise ValueError("Image is empty or too small to process.")
    if float(np.std(gray)) < 2.0:
        raise ValueError("Image is blank or has too little contrast to recognize a handshape.")
    if gray.shape == (INPUT_SIZE, INPUT_SIZE):
        resized = gray
    else:
        cropped = _largest_foreground(gray)
        height, width = cropped.shape
        scale = (INPUT_SIZE - 4) / max(height, width)
        new_width = max(1, min(INPUT_SIZE - 4, round(width * scale)))
        new_height = max(1, min(INPUT_SIZE - 4, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        hand = cv2.resize(cropped, (new_width, new_height), interpolation=interpolation)
        resized = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        x0, y0 = (INPUT_SIZE - new_width) // 2, (INPUT_SIZE - new_height) // 2
        resized[y0 : y0 + new_height, x0 : x0 + new_width] = hand

    return resized.astype(np.float32)[None, :, :, None] / 255.0
