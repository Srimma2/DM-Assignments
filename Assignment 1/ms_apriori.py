import sys,os,re,operator,itertools
from collections import defaultdict

from parser import *

#TODO - normalize the counts, 2nd level candidate gen should take argument L
# make sure that everything is passed into functions is sorted according to MIS values

def lvl2_candidategen(itemset_dict,phi,param_dict):
    c2 = [] #output
    items = sorted(itemset_dict.items(), key=operator.itemgetter(1)) #sorted list of tuples of [(key,value)....]
    setLen = len(items)
    for i in range(0,setLen): #each item i in L
        if items[i][1] >= param_dict['MIS'][items[i][0]]:
            for h in range(i+1,setLen): # each item after i in L
                if items[h][1] >= param_dict['MIS'][items[i][0]] and abs(items[h][1]-items[i][1]) <= phi:
                    candidate = [items[i][0],items[h][0]] # {i,h}
                    c2.append(candidate)
    
    return c2
    
def MS_candidategen(f,phi,param_dict):
    ck = []
    setLen = len(f) #length of f
    eLen = len(f[0])-1 #length of the element
    for i in range(0,setLen): #each item i in L
            for h in range(i+1,setLen): #each item h after i
                if f[i][:eLen] == f[h][:eLen] and (abs(param_dict['MIS'][f[i][eLen]]-param_dict['MIS'][f[h][eLen]]) <= phi):
                    if param_dict['MIS'][f[i][eLen]] < param_dict['MIS'][f[h][eLen]]:
                        c = list(f[i])
                        c.append(f[h][eLen])
                    else:
                        c = list(f[h])
                        c.append(f[i][eLen])
                    ck.append(c)
                    subsets = list(itertools.combinations(c,eLen)) #generate subsets of k-1
                    for s in subsets:
                        if c[0] in s or param_dict['MIS'][c[1]] == param_dict['MIS'][c[0]]:
                            if s not in f:
                                ck.remove(c) 
    return ck

def MS_Apriori(transaction_db,param_dict):
    """
        MS Apriori Algorithm
        Input : 
            transaction_db : transaction database
            param_dict : paramter dictionary 
                        {MIS,SDC,cannot_be_together,must_have}
        Returns frequent_itemsets : dict
    """
    # contains all itemset with their count initialized to 0; key = itemset(tuple), value = list
    itemset_dict = defaultdict(lambda : 0.0)
    # lists frequent itemsets (list) at each level; key = F_<level>, value = list
    frequent_itemsets = dict()
    # lists candidates (list) at each level; key = C_<level>, value = list
    candidate_itemsets = dict()

    # support difference constraint
    phi = param_dict['SDC']

    # get items sorted by MIS
    M = sorted(param_dict['MIS'], key = param_dict['MIS'].get)

    # size of transaction
    n = float(len(transaction_db))

    for t in transaction_db:
        for item in M:
            if item in t:
                itemset_dict[item] += 1/n

    # frequent itemset
    frequent_itemsets['F_1'] = [[key] for key,val in itemset_dict.iteritems() if val > param_dict['MIS'][key]]

    k = 2

    while(frequent_itemsets['F_'+str(k-1)]):
        if k == 2:
            # remove items with support less than min of MIS
            L = {key:val for key,val in itemset_dict.iteritems() if val > param_dict['MIS'][M[0]]}
            # list of candidates(list)
            candidate_itemsets['C_'+str(k)] = lvl2_candidategen(L,phi,param_dict)
            # print candidate_itemsets['C_' + str(k)]
        else:
            candidate_itemsets['C_'+str(k)] = MS_candidategen(frequent_itemsets['F_'+str(k-1)],phi,param_dict)

        for t in transaction_db:
            for c in candidate_itemsets['C_'+str(k)]:
                # if c is contained in t
                if set(c) < set(t):
                    # normalize
                    itemset_dict[tuple(c)] += 1/n
                # if c without first element is contained in t
                # for rule generation
                # if set(c[1:]) < set(t):
                #     itemset_dict[tuple(c[1:])] += 1/n

        # sort based on MIS
        frequent_itemsets['F_'+str(k)] = [sorted(c,key = param_dict['MIS'].get) for c in candidate_itemsets['C_'+str(k)] if itemset_dict[tuple(c)] >= param_dict['MIS'][c[1]]]
        k += 1

    return frequent_itemsets


if __name__ == "__main__":
    transaction_db = parse_input('input-data.txt')
    param_dict = parse_parameter('parameter-file.txt')
    print MS_Apriori(transaction_db,param_dict)
