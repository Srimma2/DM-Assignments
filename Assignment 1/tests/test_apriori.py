import unittest,logging

from ms_apriori import *
from parser import *

class Test_Apriori(unittest.TestCase):
    def setUp(self):
        self.transaction_db = parse_input('tests/input-data.txt')
        self.param_dict = parse_parameter('tests/parameter-file.txt')

    def test_apriori(self):
        frequent_itemsets = MS_Apriori(self.transaction_db, self.param_dict)
        self.assertEqual(1,2)

