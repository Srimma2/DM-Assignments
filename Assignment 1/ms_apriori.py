import sys,os,re
from collections import defaultdict


def MS_Apriori(transaction_db,param_dict,phi):
    """
        MS Apriori Algorithm
        Input : 
            transaction_db : transaction database
            param_dict : paramter dictionary 
                        {MIS,SDC,cannot_be_together,must_have}
            phi : difference constraint
        Returns frequent_itemsets : dict
    """
    # contains all itemset with their count initialized to 0; key = itemset(tuple), value = list
    itemset_dict = defaultdict(lambda : 0)
    # lists frequent itemsets (list) at each level; key = F_<level>, value = list
    frequent_itemsets = dict()
    # lists candidates (list) at each level; key = C_<level>, value = list
    candidate_itemsets = dict()

    # get items sorted by MIS
    M = sorted(param_dict['MIS'], key = param_dict['MIS'].get)

    # size of transaction
    n = len(transaction_db)

    # initial pass
    # count occurences  
    for key in M.keys():
        count = 0
        for t in transaction_db:
            if key in t:
                count += 1
                continue
        # convert key to tuple
        itemset_dict[tuple(key)] = count

    # frequent itemset
    frequent_itemsets['F_1'] = [list(key) for key,val in itemset_dict.iteritems() if val > param_dict['MIS'][key[0]]]

    k = 2

    while(frequent_itemsets['F_'+str(k-1)]):
        if k == 2:
            # list of candidates(list)
            candidate_itemsets['C_'+str(k)] = lvl2_candidategen(itemset_dict,phi)
        else:
            candidate_itemsets['C_'+str(k)] = MS_candidategen(frequent_itemsets['F_'+str(k-1)],phi)

        for t in transaction_db:
            for c in candidate_itemsets['C_'+str(k)]:
                # if c is contained in t
                if set(c) < set(t):
                    itemset_dict[tuple(c)] += 1
                # if c without first element is contained in t
                if set(c[1:]) < set(t):
                    itemset_dict[tuple(c[1:])] += 1

        frequent_itemsets['F_'+str(k)] = [c for c in candidate_itemsets['C_'+str(k)] if itemset_dict[tuple(c)]/n >= param_dict['MIS'][c[1:]]]
        k += 1

    return frequent_itemsets


    
