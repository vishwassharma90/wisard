# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import struct
import random
import matplotlib
#import tensorflow




def discriminator():
    tt=0
    discriminator = []
    for i in range(10):  #10 
        #print(1)
        ram = []
        for j in range((int)((nodes))): #98    
             
            total_pos = []
            total_pos1 = []
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
            total_pos1.append(positions)
            total_pos = np.vstack(positions)
            #print(total_pos)
            
            
            table = []
            max = len("{0:b}".format(2**len(total_pos))) - 1
            for i in range(2**len(total_pos)):
                x = (('0' * max) + "{0:b}".format(i))
                x = x[len(x)-max:]
                table.append({x: 0})    
            
#            if tt < 1:
#                print(table)
                
            ram.append(table)
            
        discriminator.append(ram)
#    if tt <1:
#        print(discriminator)
#        tt = tt + 1
    return discriminator
        
        
def train_discriminator(x_train, y_train):
    
    images = x_train
    lable = y_train    
    
    for i in images:
        l = lable[i]
        num = int(l)  #have to check how to do it after i do the preprocessing part and see the dataset lable type
        dis = d[num]
        for r in dis:     #here i want to iterate the ram of the perticular discriminator
            pattern_to_con = total_pos[r]    # here i want to take only the positions of the node of that discriminator
                
    
    return 1

def test_discriminator(x_test):
    
    return 1



if __name__ == "__main__":

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
        
    d = discriminator()
    d.train_discriminator(x_train,y_train)
    print(d[0][0][0])
#    train_discriminator(x_train,y_train)       