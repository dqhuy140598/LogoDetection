from unittest import TestCase
from dataset import Dataset

class TestDataset(TestCase):
    def test_parse_line(self):
        dataset = Dataset('','')
        line = '0 /content/openlogo/JPEGImages/ANZ_sportslogo_80.jpg 650 366 0 38 138 314 260 0 526 154 640 324'
        result = dataset.parse_line(line)
        print(result)
        self.assertEqual(len(result),6)
