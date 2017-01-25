import unittest

from parser import *


class Test_Parser(unittest.TestCase):

    def setUp(self):
        self.transaction_db = parse_input('tests/input-data.txt')
        self.param_dict = parse_parameter('tests/parameter-file.txt')

    def test_parse_input(self):
        ans = [[20,30,40],[20,10,30]]
        self.assertEqual(self.transaction_db,ans)

    def test_switch_parameters_MIS(self):
        key,val = switch_parameters('MIS(5) = 0.23')
        self.assertEqual(key,5)
        self.assertEqual(val,0.23)

    def test_switch_parameters_SDC(self):
        key,val = switch_parameters('SDC = 0.2')
        self.assertEqual(key,'SDC')
        self.assertEqual(val,0.2)

    def test_switch_parameters_cbt(self):
        key,val = switch_parameters('cannot_be_together: {2, 4}, {10, 30}')
        self.assertEqual(key,'cannot_be_together')
        self.assertEqual(val,[[2,4],[10,30]])

    def test_switch_parameters_mh(self):
        key,val = switch_parameters('must-have: 10 or 5 or 30')
        self.assertEqual(key,'must_have')
        self.assertEqual(val,[10,5,30])
    
    def test_parse_parameter(self):
        ans = {'MIS' : {10:0.43, 20 : 0.30, 30: 0.41, 40 : 0.22},'SDC' : 0.1, 'cannot_be_together' : [[20,30],[10,40]], 'must_have' : [20,40]}
        self.assertEqual(self.param_dict,ans)
