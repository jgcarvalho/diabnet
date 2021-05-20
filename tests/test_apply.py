import unittest
import sys

sys.path.append("../")
import diabnet.apply as apply
from diabnet.apply import encode_features


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.feat_names = [
            "snp_1126",
            "snp_2444",
            "snp_1",
            "snp_1757",
            "snp_9",
            "snp_11",
            "snp_2454",
            "snp_10",
            "snp_376",
            "snp_8",
            "snp_715",
            "snp_1417",
            "snp_1485",
            "snp_4",
            "snp_2",
            "snp_3",
            "snp_6",
            "snp_5",
            "snp_469",
            "snp_1208",
            "snp_1780",
            "snp_790",
            "snp_1301",
            "snp_2208",
            "snp_966",
            "snp_992",
            "snp_577",
            "snp_1938",
            "snp_1494",
            "snp_2478",
            "snp_1843",
            "snp_2241",
            "snp_1045",
            "snp_368",
            "snp_1652",
            "snp_299",
            "snp_615",
            "snp_1150",
            "snp_1479",
            "snp_1461",
            "snp_234",
            "snp_1166",
            "snp_7",
            "snp_1283",
            "snp_1773",
            "snp_1272",
            "snp_359",
            "snp_954",
            "snp_2371",
            "snp_1509",
            "snp_2474",
            "snp_2472",
            "snp_2470",
            "snp_2469",
            "snp_2462",
            "snp_2459",
            "snp_2458",
            "snp_2450",
            "snp_2449",
            "snp_2442",
            "snp_2439",
            "snp_2434",
            "snp_2432",
            "snp_2431",
            "snp_2430",
            "snp_2428",
            "snp_2427",
            "snp_2420",
            "snp_2419",
            "snp_2417",
            "snp_2412",
            "snp_2405",
            "snp_2402",
            "snp_2399",
            "snp_2395",
            "snp_2384",
            "snp_2377",
            "snp_2375",
            "snp_2369",
            "snp_2368",
            "snp_2362",
            "snp_2358",
            "snp_2357",
            "snp_2354",
            "snp_2352",
            "snp_2349",
            "snp_2348",
            "snp_2337",
            "snp_2326",
            "snp_2323",
            "snp_2318",
            "snp_2317",
            "snp_2316",
            "snp_2315",
            "snp_2309",
            "snp_2305",
            "snp_2302",
            "snp_2292",
            "snp_2282",
            "snp_2275",
            "AGE",
        ]

        self.predictor = apply.Predictor(
            "models/model-layer01-FEAT-SELECT-50BW-001.pth",
            self.feat_names,
            "data/negatives_older60.csv",
        )

    def test_negatives_at_age(self):
        p = self.predictor.negatives(age=70, bmi=30)
        print(p)

    def test_negatives_lifetime(self):
        p = self.predictor.negatives_life()
        print(p)


if __name__ == "__main__":
    unittest.main()
