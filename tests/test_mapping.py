import unittest

from utils.class_mapping import (
    CLASSES,
    DATASET_LABEL_TO_CLASS,
    encode_dataset_labels,
)


class ClassMappingTests(unittest.TestCase):
    def test_dataset_labels_skip_j_and_map_to_contiguous_classes(self):
        self.assertEqual(len(CLASSES), 24)
        self.assertEqual(DATASET_LABEL_TO_CLASS[8], "I")
        self.assertEqual(DATASET_LABEL_TO_CLASS[10], "K")
        self.assertEqual(DATASET_LABEL_TO_CLASS[24], "Y")
        self.assertEqual(encode_dataset_labels([8, 10, 24]), [8, 9, 23])

    def test_rejects_j_dataset_label(self):
        with self.assertRaises(ValueError):
            encode_dataset_labels([9])


if __name__ == "__main__":
    unittest.main()
