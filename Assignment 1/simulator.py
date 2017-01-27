import re,sys,os
import numpy as np
import random

def generate_parameters(seed = None):
    """
        Generates parameters
    """
    # generate param_dict
    param_dict_gen = dict()
    param_dict_gen['MIS'] = dict()

    # random seed
    random.seed(seed)

    # number of items
    item_num = random.randint(1,10)
    # list of items with their numbers, sample without replacement
    items = random.sample(range(1,1000),item_num)
    # generate MIS 
    MIS = [random.random() for _ in xrange(item_num)] 
    for key,val in zip(items,MIS):
        param_dict_gen['MIS'][key] = val

    # generate SDC
    param_dict_gen['SDC'] = random.random()

    # generate cannot_be_together
    cannot_be_together_num = random.randint(0,5)
    param_dict_gen['cannot_be_together'] = list()
    for i in range(cannot_be_together_num):
        param_dict_gen['cannot_be_together'].append(random.sample(items,random.randint(1,item_num))) 

    # generate must_have
    must_have_num = random.randint(0,5)
    param_dict_gen['must_have'] = random.sample(items,must_have_num)

    return param_dict_gen,items

def generate_transactions(items):
    """
        Generates transactions
    """
    transaction_db = list()
    n = random.randint(3,20)
    item_num = len(items)

    for _ in range(n):
        # size of transaction
        m = random.randint(1,item_num)
        transaction = random.sample(items,m)
        transaction_db.append(transaction)

    # uniqueness of transaction
    transaction_db = [list(x) for x in set(tuple(transaction) for transaction in transaction_db)]

    return transaction_db

def generate_parameter_file(param_dict):
    """
        Generate parameter file
    """
    filename = 'parameter-file.txt'

    with open(filename,'wb') as f:

        # write MIS
        for key,val in param_dict['MIS'].iteritems(): 
            f.write('MIS(' + str(key) + ') = ' + str(val) + '\n')

        # write SDC
        f.write('SDC = ' + str(param_dict['SDC']) + '\n')

        # write cannot_be_together
        f.write('cannot_be_together: ')
        tmp_list = list()
        for i_list in param_dict['cannot_be_together']:
            # replace [] with {}
            tmp = str(i_list)
            tmp = tmp.replace('[','{')
            tmp = tmp.replace(']','}')
            tmp_list.append(tmp)
        f.write(', '.join(tmp_list) + '\n')

        # write must_have
        tmp = [str(i) for i in param_dict['must_have']]
        f.write('must-have: ' + ' or '.join(tmp))

    return

def generate_input_file(transaction_db):
    """
        Generate input file
    """
    filename = 'input-data.txt'

    with open(filename,'wb') as f:
        for transaction in transaction_db:
            # replace [] with {}
            list_rep = str(transaction)
            list_rep = list_rep.replace('[','{')
            list_rep = list_rep.replace(']','}')
            # write transaction
            f.write(list_rep + '\n')

    return

if __name__ == '__main__':
    param_dict,items = generate_parameters(0)
    transaction_db = generate_transactions(items)
    generate_parameter_file(param_dict)
    generate_input_file(transaction_db)

