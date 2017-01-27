import unittest
from simulator import *

class Simulator_Test(unittest.TestCase):
    """
        Test for simulator
    """
    def setUp(self):
        self.param_dict,self.items = generate_parameters(0)
        self.transaction_db = generate_transactions(self.items)

    def test_param_dict_MIS(self):
        pass

    def test_param_dict_SDC(self):
        pass

    def test_param_dict_cbc(self):
        pass

    def test_param_dict_mh(self):
        pass

    def test_transaction_unique(self):
        pass

    def test_item_in_transaction(self):
        pass

