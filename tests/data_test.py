import sys
import unittest
import random

sys.path.append('../')
import diabnet.data as data

class TestDiabDataset(unittest.TestCase):
    def test_n_feat(self):
        c = ['snp_{}'.format(i+1) for i in range(100)]
        c.append('AGE')
        d = data.DiabDataset('../datasets/train.csv',c , label='T2D', random_age=False)
        self.assertEqual(len(c), d.n_feat)

    def test_len_dataset(self):
        c = ['snp_{}'.format(i+1) for i in range(100)]
        c.append('AGE')
        d = data.DiabDataset('../datasets/train.csv',c , label='T2D', random_age=False)
        self.assertEqual(len(d), 1638)

    def test_age_bmi(self):
        c1 = ['snp_1', 'AGE', 'BMI', 'snp_2']
        d1 = data.DiabDataset('../datasets/train.csv',c1 , label='T2D', random_age=True)
        c2 = ['AGE', 'snp_1', 'snp_2','BMI']
        d2 = data.DiabDataset('../datasets/train.csv',c2 , label='T2D', random_age=False)
        r = random.randint(0, len(d1))
        print(d1[r])
        print(d2[r])
        self.assertEqual(d1[3][0][0,1].numpy(), d2[3][0][0,0].numpy())
        self.assertEqual(d1[3][0][0,2].numpy(), d2[3][0][0,3].numpy())

if __name__ == "__main__":
    unittest.main()