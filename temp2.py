# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import struct
import random

# the mnist has 28*28(784) pixels that we need to train in the discriminator
# we have to initialize 10 discriminator for 0 to 9 number images(one discriminator for one pattern)
# We are taking 
#

input_size = 28*28
no_of_rand_pix_selec = 8   
nodes = input_size/no_of_rand_pix_selec    #98
#print(nodes)
#n = random.randint(1, input_seize)
#print(n)

def discriminator():
    tt=0
    for i in range(10):  #10
        discriminator = []
        #print(1)
        for j in range((int)((nodes))): #98    
            ram = [] 
            total_pos = []
            positions = []
            for k in range(no_of_rand_pix_selec):  #8
                
                n_0 = [0]
                n = random.randint(1, input_size)
                
                for i in n_0:
                    if n != i:
                        positions.append(n)    
            
                n_0.append(n)
            
            #if tt < 1:
            #    print(positions)
            #    tt = tt + 1
            
            total_pos = np.vstack(positions)
            #print(total_pos)
            
            
            table = []
            max = len("{0:b}".format(2**len(total_pos))) - 1
            for i in range(2**len(total_pos)):
                x = (('0' * max) + "{0:b}".format(i))
                x = x[len(x)-max:]
                table.append({x: 0})    
            
           # if tt < 2:
           #     print(table)
           #     tt = tt + 1
            
        discriminator.append(table)    
        if tt == 0:
            print(discriminator)    
            tt = tt + 1

if __name__ == "__main__":

    discriminator()       