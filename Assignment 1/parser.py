import os,sys
import re

def parse_input(filename = 'input-data.txt'):
    """
        Parses input file
        Input: filename
        Returns transaction database 
    """
    with open(filename) as f:
        contents = f.readlines()

    transaction_db = list()

    remove_brackets = re.compile(r'{|}')

    for content in contents:
        # removes brackets
        content = remove_brackets.sub('',content)
        content = content.split(',')
        # convert to int
        content = [int(i) for i in content]
        transaction_db.append(content)

    return transaction_db


def switch_parameters(x):
    """
        Select parameter key 
        Return key and value
    """
    if re.search('MIS.+',x):
        tmp = re.search('MIS\((.+)\) = (.+)',x)
        key,val = int(tmp.group(1)),float(tmp.group(2))
    elif re.search('SDC.+',x):
        key = 'SDC'
        val = re.search('SDC = (.+)',x).group(1)
        val = float(val)
    elif re.search('cannot_be_together.+',x):
        key = 'cannot_be_together'
        tmp = re.search('cannot_be_together: (.+)',x).group(1)
        tmp = re.findall('{(\d+, \d+)}',tmp)
        val = [i.split(',') for i in tmp]
        # convert to int
        val = [[int(j) for j in i] for i in val]

    elif re.search('must-have.+',x):
        key = 'must_have'
        tmp = re.search('must-have: (.+)',x).group(1)
        val = re.findall('\d+',tmp)
        val = [int(i) for i in val]

    return key,val



def parse_parameter(filename = 'parameter-file.txt'):
    """
        Parses parameter file
        Input : Filename
        Output : 
    """
    with open(filename) as f:
        contents = f.readlines()

    param_dict = dict()
    param_dict['MIS'] = {}

    for content in contents:
        key,value = switch_parameters(content) 
        if isinstance(key,int):
            key = key
            param_dict['MIS'][key] = value
        else:
            param_dict[key] = value

    return param_dict


# if __name__ == "__main__":
#     transaction_db = parse_input()
#     parse_parameter()

