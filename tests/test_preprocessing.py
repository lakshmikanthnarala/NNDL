import unittest

import numpy as np
from PIL import Image

from preprocessing import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_28_by_28_image_is_normalized_and_batched(self):
        pixels = np.arange(784, dtype=np.uint8).reshape((28, 28))
        image = Image.fromarray(pixels)
        tensor = preprocess_image(image)
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertTrue(np.allclose(tensor[0, :, :, 0], pixels / 255.0))

    def test_rejects_blank_image(self):
        image = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "blank"):
            preprocess_image(image)


if __name__ == "__main__":
    unittest.main()
