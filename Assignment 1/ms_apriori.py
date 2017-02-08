import sys,os,re,operator,itertools
from collections import defaultdict

from parser import *


def lvl2_candidategen(itemset_dict,phi,param_dict,n):
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
    
def MS_candidategen(f,phi,param_dict,itemset_dict,n):
    ck = []
    setLen = len(f) #length of f
    eLen = len(f[0])-1 #length of the element
    for i in range(0,setLen): #each item i in L
        for h in range(i+1,setLen): #each item h after i
            if f[i][:eLen] == f[h][:eLen] and (abs(itemset_dict[f[i][eLen]]-itemset_dict[f[h][eLen]]) <= phi):
                if itemset_dict[f[i][eLen]] < itemset_dict[f[h][eLen]]:
                    c = list(f[i])
                    c.append(f[h][eLen])
                else:
                    c = list(f[h])
                    c.append(f[i][eLen])
                ck.append(c)
                subsets = list(itertools.combinations(c,eLen)) #generate subsets of k-1
                for s in subsets:
                    s = list(s)
                    if (c[0] in s) or (param_dict['MIS'][c[1]] == param_dict['MIS'][c[0]]):
                        if s < f:
                            pass
                        else:
                            ck.remove(c) 
    return ck

def post_processing(f,param_dict):
    """
        Processes must-have and cannot be together
        Input : 
            f : frequent itemset
            param_dict : parameter dictionary
    """
    must_have = param_dict['must_have']
    cannot_be_together = param_dict['cannot_be_together']

    # must have 
    tmp  = list()
    for itemset in f:
        if set(itemset).intersection(set(must_have)): 
            tmp.append(itemset)

    f = tmp

    # cannot be together
    tmp = f
    for itemset in f:
        for cbt in cannot_be_together:
            if cbt in itemset:
                tmp.remove(itemset)

    return tmp

def MS_Apriori(transaction_db,param_dict):
    """
        MS Apriori Algorithm
        Input : 
            transaction_db : transaction database
            param_dict : paramter dictionary 
                        {MIS,SDC,cannot_be_together,must_have}
        Returns frequent_itemsets : dict
    """
    # contains all itemset with their count initialized to 0; key = itemset(tuple), value = count
    itemset_dict = defaultdict(lambda : 0.0)
    # does tail count; key = itemset(tuple), value = count
    tail_count = defaultdict(lambda : 0.0)
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
                itemset_dict[item] += 1

    # frequent itemset
    frequent_itemsets['F_1'] = [[key] for key,val in itemset_dict.iteritems() if val/n > param_dict['MIS'][key]]

    k = 2

    while(frequent_itemsets.get('F_'+str(k-1))):
        if k == 2:
            # remove items with support less than min of MIS
            L = {key:val for key,val in itemset_dict.iteritems() if val/n > param_dict['MIS'][M[0]]}
            print "L is {}".format(L)
            # list of candidates(list)
            candidate_itemsets['C_'+str(k)] = lvl2_candidategen(L,phi,param_dict,n)
        else:
            candidate_itemsets['C_'+str(k)] = MS_candidategen(frequent_itemsets['F_'+str(k-1)],phi,param_dict,itemset_dict,n)

        if not candidate_itemsets['C_' + str(k)]:
            break

        print "candidate C_{} is {}".format(k,candidate_itemsets['C_' + str(k)])
        for t in transaction_db:
            for c in candidate_itemsets['C_'+str(k)]:
                # if c is contained in t
                if set(c) < set(t):
                    # normalize
                    itemset_dict[tuple(c)] += 1
                # perform tail count, used for rule generation
                if set(c[1:]) < set(t):
                    tail_count[tuple(c)] += 1

        # sort based on MIS
        ans = [sorted(c,key = param_dict['MIS'].get) for c in candidate_itemsets['C_'+str(k)] if itemset_dict[tuple(c)]/n >= param_dict['MIS'][c[0]]]

        if ans:
            frequent_itemsets['F_' + str(k)] = ans

        k += 1

    # frequent_itemsets = {key : value for key,value in frequent_itemsets.iteritems() if value}

    # post processing
    for k,v in frequent_itemsets.iteritems():
        if v:
            frequent_itemsets[k] = post_processing(v,param_dict)


    return frequent_itemsets,tail_count


if __name__ == "__main__":
    transaction_db = parse_input('input-data.txt')
    param_dict = parse_parameter('parameter-file.txt')
    print transaction_db
    print param_dict
    print MS_Apriori(transaction_db,param_dict)
