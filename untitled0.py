#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:14:54 2019

@author: iss
"""

from numba import njit,jit,vectorize
from numba import cuda
import numpy as np
#import struct
import random
import time

@vectorize(['int32(int32)'], target = 'cuda')
def addition(d):
    for i in range(50000000):
        i +=1
    return d
 
    


if __name__ == "__main__":
    
    d0 = [1,2,3,6357,467,467,54]
    
#    d = [[[{'23':1 , '2':32},{'24':8 , '35':2}],[[647,5],[456,5],[35,4]]]]
    
    my_dict = [[{'00001' : 1, '24324' : 0},{'13' : 1, '24' : 0}],[{'13124' : 1, '24324' : 0},{'13' : 1, '24' : 0}]]
    di = []
    for j in range(len(my_dict)):
        for i in range(len(my_dict[j])):
            for key, value in my_dict[j][i].items():
                temp = [key,value]
                di.append(temp)
    
    di = np.asarray(di).astype(np.int32)
    
    
    starttest = time.time()
    addi = addition(di)
    endtest = time.time()
    print("time test = ",endtest - starttest)
    print(addi)
    
#    print(type(d0))
#    print(type(d))
#    d = np.asarray(d).astype(np.int32)
#    #print(type(d[0][0][0][0]))
#    addi = addition(d)
#    print(addi)



