"""Run after training to verify model reload and prediction output."""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from predict import DEFAULT_MODEL, load_predictor, predict_image
from utils.class_mapping import CLASSES


class ModelSmokeTests(unittest.TestCase):
    def test_saved_model_loads_and_predicts(self):
        if not DEFAULT_MODEL.is_file():
            self.skipTest("Train the model first with `python train_model.py`.")
        model = load_predictor(DEFAULT_MODEL)
        image = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
        letter, confidence, processed = predict_image(image, model)
        self.assertIn(letter, CLASSES)
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertEqual(processed.shape, (28, 28))


if __name__ == "__main__":
    unittest.main()
