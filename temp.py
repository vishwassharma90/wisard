# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import struct
import random
#import matplotlib
#import tensorflow                          #i have tensorflow in the system, the tensorflow environment is also there
                                                     #i am able to activate it in terminal by (conda activate tensorflow_gpuenv)
                                                     #i am able to import it in gedit python file after that          
                                                     #but i am not able to import it in the anacoda(in this file)

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result




def discriminator():
    tt=0
    discriminator = []
    accumulated_pos = []
    for i in range(10):  #10 
        #print(1)
        ram = []
        
        for j in range((int)((nodes))): #98    
             
            total_pos = []
            total_pos1 = []
            positions = []
            for k in range(no_of_rand_pix_selec):  #8
                n_0 = [0]                               # this is wrong as for every ram in a discriminator this will reset
                n = random.randint(1, input_size)                     #and we want it to reset after 98 ram of one discriminator
                                                                          #but computer hange if i do it. so check later
                for i in n_0:
                    if n != i:
                        positions.append(n)    
            
                n_0.append(n)
                
            #if tt < 1:
            #    print(positions)
            #    tt = tt + 1
            accumulated_pos.append(positions)
            total_pos = np.vstack(positions)
            #print(total_pos)
            
            table = []
            dictionary = {}
            max = len("{0:b}".format(2**len(total_pos))) - 1
            for i in range(2**len(total_pos)):
                x = (('0' * max) + "{0:b}".format(i))
                x = x[len(x)-max:]
                dictionary[x] = 0
                #table.append({x: 0})    
            table.append(dictionary)
            
#            if tt < 1:
#                print(table)
                
            ram.append(table)
            
        discriminator.append(ram)
#    if tt <1:
#        print(discriminator)
#        tt = tt + 1
    #print(accumulated_pos[0])
    return discriminator, accumulated_pos
        
        
def train_discriminator_with_bleaching(d,pos,x_train, y_train):
    
    images = x_train
    lable = y_train    
    
    for i,image in enumerate(images):
        l = lable[i]
        num = l  #have to check how to do it after i do the preprocessing part and see the dataset lable type
        all_ram_of_selected_discriminator = d[num]
        #print(all_ram)
        t_ratina = pos[(98*num):(98*num+98)]
        #print(t_ratina)
        for x,r in enumerate(all_ram_of_selected_discriminator):     #here i want to iterate the ram of the perticular discriminator
            #print(x,r)
            #pattern_to_con = total_pos[r]    # here i want to take only the positions of the node of that discriminator
            #print(x,r)
            ratina_for_one_ram = t_ratina[x]
            #print(ratina_for_one_ram)
            #for i in (0,len(ratina_for_one_ram)):
             #   print(i)
            threshold = 0         # this define the threshold. right now if any one pixel in ratina(8) is >=1 then the value
            n = []                                                              #saved is 1. if we want to chnage it we can     
            for pix in ratina_for_one_ram:                                          #like if value of 5 pixel is>=1 then save 1
                if image[(pix-1)]>=1:
                    n.append(1)
                    threshold = threshold + 1
                else:
                    n.append(0)
            #print(n)
            
            address_of_that_ram = concatenate_list_data(n)
            print(address_of_that_ram)
            #print(threshold)
            
            if threshold >= 1:                  #refer above comment
                for index,key in enumerate(r[0]):
                    #print(1)
                    #print(key)
                    if key == address_of_that_ram:
                        r[0][key] += 1
                        #print(2)
#                    r[address_of_that_ram] += 1
#                    print(address_of_that_ram)
#                    print(r[address_of_that_ram])
            else:
                print(0)
            print(x,r)
            #address_of_one_ram = ','.join(n)
            #print(address_of_one_ram)
    
    return 1

def test_discriminator_with_bleaching(d,acc_pos,x_test,y_test):
    
    
    
    
    
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
    
    x_train1 = np.random.randint(10, size=784)
    x_train = [x_train1]
    y_train = [0,1,2,3,4,5,6,7,8,9]    
    
    
    x_test1 = np.random.randint(10, size=784)
    x_test = [x_test1]
    y_test = [0,1,2,3,4,5,6,7,8,9]
    
    
    d = []
    d, acc_pos = discriminator()
    #print(d[0][0])
    #print(acc_pos)
    #print(x_train)
    train_the_network = train_discriminator_with_bleaching(d,acc_pos,x_train,y_train)
    #print(view)
    test_the_network = test_discriminator_with_bleaching(d,acc_pos,x_test,y_test)       