import os
import pickle
import unittest
import random
import diabnet.data as data

TEST_FILE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestData(unittest.TestCase):
    def test_get_feature_names(self):
        # Get example data
        fn = os.path.join(TEST_FILE_DIR, "example_data.csv")
        # Default arguments
        result = data.get_feature_names(fn)
        with open(os.path.join(TEST_FILE_DIR, "feature_names.data"), "rb") as file:
            expected = pickle.load(file)
        self.assertListEqual(result, expected)
        # use_sex = False
        result = data.get_feature_names(fn, use_sex=False)
        expected.remove("sex")
        self.assertListEqual(result, expected)
        # use_parents_diagnosis = False
        result = data.get_feature_names(fn, use_parents_diagnosis=False)
        with open(os.path.join(TEST_FILE_DIR, "feature_names.data"), "rb") as file:
            expected = pickle.load(file)
        for feature in ["mo_t2d", "fa_t2d"]:
            expected.remove(feature)
        self.assertListEqual(result, expected)


# class TestDiabDataset(unittest.TestCase):
#     def test_n_feat(self):
#         c = ["snp_{}".format(i + 1) for i in range(100)]
#         c.append("AGE")
#         d = data.DiabDataset("../datasets/train.csv", c, label="T2D", random_age=False)
#         self.assertEqual(len(c), d.n_feat)

#     def test_len_dataset(self):
#         c = ["snp_{}".format(i + 1) for i in range(100)]
#         c.append("AGE")
#         d = data.DiabDataset("../datasets/train.csv", c, label="T2D", random_age=False)
#         self.assertEqual(len(d), 1638)

#     def test_age_bmi(self):
#         c1 = ["snp_1", "AGE", "BMI", "snp_2"]
#         d1 = data.DiabDataset("../datasets/train.csv", c1, label="T2D", random_age=True)
#         c2 = ["AGE", "snp_1", "snp_2", "BMI"]
#         d2 = data.DiabDataset(
#             "../datasets/train.csv", c2, label="T2D", random_age=False
#         )
#         r = random.randint(0, len(d1))
#         print(d1[r])
#         print(d2[r])
#         self.assertEqual(d1[3][0][0, 1].numpy(), d2[3][0][0, 0].numpy())
#         self.assertEqual(d1[3][0][0, 2].numpy(), d2[3][0][0, 3].numpy())


if __name__ == "__main__":
    unittest.main()
