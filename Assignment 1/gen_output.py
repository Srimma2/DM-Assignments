from ms_apriori import *
from parser import *
import re,sys


def gen_output(input_file = 'input-data.txt',parameter_file = 'parameter-file.txt',output_file = 'output-patterns.txt'):
    """
        Output generator
    """
    transaction_db = parse_input(input_file)
    param_dict = parse_parameter(parameter_file)
    frequent_itemsets,tail_count,itemset_dict = MS_Apriori(transaction_db,param_dict)
    with open(output_file,'w') as f:
        for key,val in frequent_itemsets.iteritems():
            if not val:
                return
            f.write('Frequent' + key.split('F_')[1] + '-itemsets')
            f.write('\n\n')
            for itemset in val:
                tmp = list()
                tmp = str(itemset)
                tmp = tmp.replace('[','{')
                tmp = tmp.replace(']','}')
                if len(itemset) == 1:
                    count = itemset_dict[itemset[0]]
                else:
                    count = itemset_dict[tuple(itemset)]
                f.write('\t' + str(count) + ':' + tmp + '\n')
                if key.split('F_')[1] != '1':
                    f.write('Tailcount = ' + str(tail_count[tuple(itemset)]))
                    f.write('\n')
        
            f.write('\n\t' + 'Total number of frequent ' + key.split('F_')[1] + '-itemsets = ' + str(len(val)))
            f.write('\n\n')


if __name__ == '__main__':
    gen_output(input_file = sys.argv[1], parameter_file = sys.argv[2],output_file = sys.argv[3])
