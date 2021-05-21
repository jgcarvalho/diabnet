import unittest
import numpy as np
from diabnet.metrics import ece, mce


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.targets = np.arange(0.0, 1.0, 0.01)
        self.predictions = np.arange(0.0, 1.0, 0.01)

    def test_ece(self):
        result = ece(self.targets, self.predictions, 10)
        self.assertEqual(result, 0.0)

    def test_mce(self):
        result = mce(self.targets, self.predictions, 10)
        self.assertEqual(result, 0.0)
