"""Correct mapping between Sign Language MNIST labels and ASL letters."""

from __future__ import annotations

from typing import Iterable


DATASET_LABELS = tuple(label for label in range(25) if label != 9)
CLASSES = tuple(chr(ord("A") + label) for label in DATASET_LABELS)
DATASET_LABEL_TO_CLASS = dict(zip(DATASET_LABELS, CLASSES))
DATASET_LABEL_TO_INDEX = {label: index for index, label in enumerate(DATASET_LABELS)}


def encode_dataset_labels(labels: Iterable[int]) -> list[int]:
    """Map original Kaggle labels, which skip 9, to contiguous model indices."""
    encoded: list[int] = []
    for raw_label in labels:
        label = int(raw_label)
        if label not in DATASET_LABEL_TO_INDEX:
            raise ValueError(f"Unexpected Sign Language MNIST label: {label}")
        encoded.append(DATASET_LABEL_TO_INDEX[label])
    return encoded


def class_for_index(index: int) -> str:
    if not 0 <= index < len(CLASSES):
        raise ValueError(f"Class index out of range: {index}")
    return CLASSES[index]
