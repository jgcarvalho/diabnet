import unittest
import numpy as np
import torch
from diabnet.metrics import ece, mce, ece_mce


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.targets = torch.arange(0.0, 1.0, 0.01)
        self.predictions = torch.arange(0.0, 1.0, 0.01)

    def test_ece(self):
        result = ece(self.targets, self.predictions, 10)
        self.assertEqual(result, 0.0)

    def test_mce(self):
        result = mce(self.targets, self.predictions, 10)
        self.assertEqual(result, 0.0)

    def test_mce_ece(self):
        result = ece_mce(self.predictions, self.targets, 10)
        self.assertTupleEqual(result, (0.0, 0.0))
