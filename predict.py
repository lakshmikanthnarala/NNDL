"""Load the trained SignSpeak model and predict a single static ASL letter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import preprocess_image
from utils.class_mapping import CLASSES, class_for_index


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "sign_language_model.keras"


def load_predictor(model_path: str | Path = DEFAULT_MODEL):
    """Load a Keras model and its persisted class mapping."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python train_model.py` first."
        )
    mapping_path = model_path.parent / "class_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Class mapping not found at {mapping_path}. Run `python train_model.py` first."
        )
    with mapping_path.open(encoding="utf-8") as file:
        mapping = json.load(file)
    if tuple(mapping.get("classes", ())) != CLASSES:
        raise ValueError("Saved class mapping does not match the Sign Language MNIST mapping.")

    try:
        from tensorflow import keras
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for predictions. Install dependencies from requirements.txt."
        ) from error
    model = keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != (28, 28, 1) or model.output_shape[-1] != len(CLASSES):
        raise ValueError("Model input/output shape is incompatible with the SignSpeak classifier.")
    return model


def predict_image(image: Image.Image | np.ndarray, model) -> tuple[str, float, np.ndarray]:
    """Return predicted letter, softmax confidence, and normalized 28 × 28 image."""
    letter, confidence, normalized, _ = predict_image_details(image, model)
    return letter, confidence, normalized


def predict_image_details(image: Image.Image | np.ndarray, model):
    """Return the prediction, normalized input, and model probabilities."""
    tensor = preprocess_image(image)
    probabilities = np.asarray(model.predict(tensor, verbose=0))
    if probabilities.shape != (1, len(CLASSES)) or not np.isfinite(probabilities).all():
        raise RuntimeError("The model returned invalid prediction probabilities.")
    index = int(np.argmax(probabilities[0]))
    confidence = float(probabilities[0, index])
    return class_for_index(index), confidence, tensor[0, :, :, 0], probabilities[0]
