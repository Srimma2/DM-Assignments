# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 01:55:35 2017

@author: Family
"""
import operator

keys = [1,2,3,4] #items
values = [10,20,5,6] #MIS values

misdic = {}
for i in range(0,len(keys)):
    misdic[keys[i]] = values[i]
    
transaction_db = [[1,1,2,3],[10,20,30,1]]

#sort dic by value - list of tuples of [(key,value)]
def dicSort(dic):
    sorted_list = sorted(dic.items(), key=operator.itemgetter(1))
    return sorted_list
    


class Transactions:
    def __init__(self,data):
        self.NoTransactions = len(data) #number of transations 
        self.db = [list(set(sublist)) for sublist in data] #remove duplicates within each Transaction i
        self.flattend = [val for sublist in data for val in set(sublist)] #flatten database for easy counting
        self.Items = list(set(self.flattend))
        
#T = transactions 
def initPass(T,M):
    output = []
    #key = number , value = frequency
    countDic = {item:1.0*T.flattend.count(item)/T.NoTransactions for item in T.Items}
    countDic = {3:6,4:3,1:9,2:25}
    sortCount = dicSort(countDic) #sort on frequency
    for i in range(0,len(sortCount)): #populate L
        if sortCount[i][1] >= M[i][1]:
            output.append(sortCount[i][0]) #insert the key
            for j in range(i+1,len(sortCount)):
                if sortCount[j][1] >= M[i][1]:
                    output.append(sortCount[j][0]) #insert the key
            break
    return output,countDic,sortCount
            
        
    
    
T = Transactions(transaction_db)   
M = dicSort(misdic) #sort the MIS values
L,countDic,sortCount = initPass(T,M)
#determine F1
F1 = []
for i in L:
    if countDic[i] >= misdic[i]:
        F1.append([i])

    

