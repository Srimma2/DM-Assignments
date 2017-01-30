import sys,os,re,operator,itertools
from collections import defaultdict

#TODO - normalize the counts, 2nd level candidate gen should take arguement L
# make sure that everything is passed into functions is sorted according to MIS values

def lvl2_candidategen(itemset_dict,phi,param_dict):
    c2 = [] #output
    items = sorted(itemset_dict.items(), key=operator.itemgetter(1)) #sorted list of tuples of [(key,value)....]
    setLen = len(items)
    for i in range(0,setLen): #each item i in L
        if items[i][1] >= param_dict['MIS'][items[i]]:
            for h in range(i+1,setLen): # each item after i in L
                if items[h][1] >= param_dict['MIS'][items[i]] and abs(items[h][1]-items[i][1]) <= phi:
                    candidate = [items[i][0],items[h][0]] # {i,h}
                    c2.append(candidate)
    
    return c2
    
def MS_candidategen(f,phi,param_dict):
    ck = []
    setLen = len(f) #lenght of f
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
            candidate_itemsets['C_'+str(k)] = lvl2_candidategen(itemset_dict,phi,param_dict)
        else:
            candidate_itemsets['C_'+str(k)] = MS_candidategen(frequent_itemsets['F_'+str(k-1)],phi,param_dict)

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

