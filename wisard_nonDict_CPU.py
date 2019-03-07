# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import struct
import time
from numba import njit,jit,vectorize
from numba import cuda
import numpy as np
#import struct
import random
#import matplotlib
import tensorflow as tf                         #i have tensorflow in the system, the tensorflow environment is also there
                                                     #i am able to activate it in terminal by (conda activate tensorflow_gpuenv)
                                                     #i am able to import it in gedit python file after that          
                                                     #but i am not able to import it in the anacoda(in this file)


#def readFiles(labelsFilePath, imagesFilePath):
#    flImg = open(imagesFilePath, 'rb')
#    flLbl = open(labelsFilePath, 'rb')
#    (mNumberImg, sizeImg, height, width) = struct.unpack('>IIII', flImg.read(16))
#    (mNumberLbl, sizeLbl) = struct.unpack('>II', flLbl.read(8))
#    imgs = []
#    labels = map(ord, flLbl.readlines()[0])
#    for i in range(sizeImg):
#        imgs.append({'img': map(ord, list(flImg.read(width*height))), 'label': labels[i]})
#    flImg.close()
#    flLbl.close()
#    return imgs

#@vectorize
#@jit
@jit                                                     
def preprocessing(tx_train, ty_train, tx_test, ty_test):
    
    py_train = ty_train.flatten()
    py_test = ty_test.flatten()
        
#    px_test = [l.tolist() for l in tx_test]
#    px_train = [l.tolist() for l in tx_train]
#    print(px_test) 
    
    px_test = tx_test.reshape(10000,784)
    px_train = tx_train.reshape(60000,784)
    #print(py_test)
    #print(px_train.shape)
    
    return px_train, py_train, px_test, py_test


#@jit
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

#@jit
def discriminator():
#    tt=0
    discriminator = []
    accumulated_pos = []
    my_list = list(range(0,784))
    for i in range(10):  #10
        #print(1)
        ram = []
        #n_0 = [1990]
#        taken = []
        random.shuffle(my_list)                                      #with this we can access every pixels of the image
#        n_0 =[]
        for j in range((int)((nodes))): #98    
             
            total_pos = []
#            total_pos1 = []
            positions = []
            
            #for k in range(no_of_rand_pix_selec):  #8
            positions = my_list[j*8:j*8+8]                            #with this we can access every pixels of the image
            #print(positions)
#            for k in range(no_of_rand_pix_selec):  #8
##                                               # this is wrong as for every ram in a discriminator this will reset
#                n = random.randint(1, input_size)                     #and we want it to reset after 98 ram of one discriminator
#                                                                          #but computer hange if i do it. so check later
#                for i in n_0:
#                    if n != i:
#                        positions.append(n)    
            
#            n_0.append(n)
#            print(n_0)
#                if taken.count(n) == 0:
#                    positions.append(n)                                                          
                #taken.append(n)                                                          
                
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
        
        
        di = []
        for j in range(len(ram)):
            for i in range(len(ram[j])):
                for key, value in ram[j][i].items():
                    temp = [key,value]
#                    print(temp)
                    di.append(temp)
        
#        print(1)
        
        discriminator.append(di)
#    if tt <1:
#        print(discriminator)
#        tt = tt + 1
    #print(accumulated_pos[0])
    return discriminator, accumulated_pos
        
#@jit
#@njit     
#@vectorize(['int32(int32,int32,int32,int32)'], target = 'cuda')
def train_discriminator_with_bleaching(d,pos,x_train, y_train):
    
    images = x_train
    lable = y_train    
    
    for i,image in enumerate(images):
#        print(image)
        l = lable[i]
        num = l  #have to check how to do it after i do the preprocessing part and see the dataset lable type
        all_ram_of_selected_discriminator = d[num]
        #print(all_ram)
        t_ratina = pos[(98*num):(98*num+98)]
        #print(t_ratina)
        
        
        for i in range(98):
            part = all_ram_of_selected_discriminator[(256*i):(256*i+256)]
            ratina_for_one_ram = t_ratina[i]
#            print(part)
            threshold = 0         # this define the threshold. right now if any one pixel in ratina(8) is >=1 then the value
            n = []                                                              #saved is 1. if we want to chnage it we can     
            for pix in ratina_for_one_ram:                                          #like if value of 5 pixel is>=1 then save 1
                if image[(pix-1)]>=1:
                    n.append(1)
                    threshold = threshold + 1
                else:
                    n.append(0)
                    
            address_of_that_ram = (int)(concatenate_list_data(n))
#            print(address_of_that_ram)
            if threshold >= 1:                  #refer above comment
                for key,index in enumerate(part):
                    #print(1)
#                    print(key,index)
#                    print(index[0])
                    if index[0] == address_of_that_ram:
                        index[1] += 1
            
        
        
#        for x,r in enumerate(all_ram_of_selected_discriminator):     #here i want to iterate the ram of the perticular discriminator
#            #print(x,r)
#            #pattern_to_con = total_pos[r]    # here i want to take only the positions of the node of that discriminator
#            #print(x,r)
#            ratina_for_one_ram = t_ratina[x]
#            #print(ratina_for_one_ram)
#            #for i in (0,len(ratina_for_one_ram)):
#             #   print(i)
#            threshold = 0         # this define the threshold. right now if any one pixel in ratina(8) is >=1 then the value
#            n = []                                                              #saved is 1. if we want to chnage it we can     
#            for pix in ratina_for_one_ram:                                          #like if value of 5 pixel is>=1 then save 1
#                if image[(pix-1)]>=1:
#                    n.append(1)
#                    threshold = threshold + 1
#                else:
#                    n.append(0)
#            #print(n)
#            
##            address_of_that_ram = concatenate_list_data(n)
#            address_of_that_ram = n
#            #print(address_of_that_ram)
#            #print(threshold)
#            
#            if threshold >= 1:                  #refer above comment
#                for key,index in enumerate(r[0][(98*num):(98*num+98)]):
#                    #print(1)
#                    #print(key)
#                    if key == address_of_that_ram:
#                        r[0][key] += 1
                        #print(key)
                        #print(index)
                        #print(2)
#                    r[address_of_that_ram] += 1
#                    print(address_of_that_ram)
#                    print(r[address_of_that_ram])
#            else:
#                print()
            #print(x,r[0])
            #address_of_one_ram = ','.join(n)
            #print(address_of_one_ram)
    
    return d
#@jit
def test_discriminator_with_bleaching(d,pos,x_test,y_test):
    right = 0
    wrong = 0
    images = x_test
    lable = y_test
    non_rec = 0
    
    for i,image in enumerate(images):
        actual_lable = lable[i]
        #sum_of_ram_output = []
#        print(image)
        total_sum=[]
#        total_sum_of_all_ram=[]
        
        for index,dis in enumerate(d):
            t_ratina = pos[(98*index):(98*index+98)]
            
            sum_of_ram_output = 0
            
            
            for i in range(98):
                part = dis[(256*i):(256*i+256)]
                ratina_for_one_ram = t_ratina[i]
                
                n = []                                                                
                for pix in ratina_for_one_ram:
                    if image[(pix-1)]>=1:
                        n.append(1)
                    else:
                        n.append(0)
                        #print(n)
            
                address_of_that_ram = (int)(concatenate_list_data(n))
                
                for key,index in enumerate(part):
                    if index[0] == address_of_that_ram and index[1]>=1:
                        #print(111)
                        #print(key,r[0][key])
                        sum_of_ram_output += 1
                
            total_sum.append(sum_of_ram_output)        
        if max(total_sum) >= 1:
            index_of_dis = total_sum.index(max(total_sum))
            if index_of_dis == actual_lable:
                right += 1
                #print(1)
            else:
                wrong += 1
                #print(0)
           
        else:
            #wrong += 1
            non_rec += 1    
            
            
            
#            for x,r in enumerate(dis):
#                ratina_for_one_ram = t_ratina[x]
#                
#                n = []                                                                
#                for pix in ratina_for_one_ram:
#                    if image[(pix-1)]>=1:
#                        n.append(1)
#                    else:
#                        n.append(0)
#                        #print(n)
#            
#                address_of_that_ram = concatenate_list_data(n)
#                #print(address_of_that_ram)
#                
#                
#                for index,key in enumerate(r[0]):
#                    if key == address_of_that_ram and r[0][key]>=1:
#                        #print(111)
#                        #print(key,r[0][key])
#                        sum_of_ram_output += 1
#                        
#                
#            total_sum.append(sum_of_ram_output)
##        print(1)
#        print(total_sum)
#            
#        if max(total_sum) >= 1:
#            index_of_dis = total_sum.index(max(total_sum))
#            if index_of_dis == actual_lable:
#                right += 1
#                #print(1)
#            else:
#                wrong += 1
#                #print(0)
#        
#        else:
#            #wrong += 1
#            non_rec += 1
    
    print(1)    
    print("non recognized images = ", non_rec)
        
    
    return right,wrong



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
    
#    imagesTrainingFile = 'train-images-idx3-ubyte'
#    labelsTrainingFile = 'train-labels-idx1-ubyte'
#    imagesTestFile = 't10k-images-idx3-ubyte'
#    labelsTestFile = 't10k-labels-idx1-ubyte'
#    
#    training = readFiles(labelsTrainingFile, imagesTrainingFile)
#    
#    for i in range(3):
#        print(training[i])
    
    x_train1 = np.random.randint(10, size=784)
    x_train2 = np.random.randint(10, size=784)
    x_train3 = np.random.randint(10, size=784)
    x_train = [x_train1,x_train2,x_train3]
    y_train = [0,1,2]    
    #print(x_train[0])
    
    
    x_test1 = np.random.randint(10, size=784)
    #print(x_test1)
    x_test2 = np.zeros(784)                    #use this for accuracy = 0 check
    x_test = [x_train1,x_test1,x_test2]                             #use this for accuracy = 100 check
#    print(x_test)
    y_test = [0,1,2]
    #print(x_test[0],x_test1)
    
    #d = []
    d, acc_pos = discriminator()
    
    d = np.asarray(d).astype(np.int32)
#    print(d)
#    
    print(type(d))
    
    acc_pos = np.asarray(acc_pos).astype(np.int32)
    print(type(acc_pos))
    
    #print(d[0][0])
    #print(acc_pos)
    #print(x_train)
    (tx_train, ty_train), (tx_test, ty_test) = tf.keras.datasets.mnist.load_data()
    px_train, py_train, px_test, py_test = preprocessing(tx_train, ty_train, tx_test, ty_test)
    
    #print(px_train)
    
#    print(px_test[0].shape)
    px_train = np.asarray(px_train).astype(np.int32)
    py_train = np.asarray(py_train).astype(np.int32)
    px_test = np.asarray(px_test).astype(np.int32)
    py_test = np.asarray(py_test).astype(np.int32)
    
    
    
    
    starttrain = time.time()
    train_the_network = train_discriminator_with_bleaching(d,acc_pos,px_train[0:1000],py_train[0:1000])
    endtrain = time.time()
    print("time train = ",endtrain - starttrain)
    #print(view)
    #print(x_test[1])
    print('this is d[0] =',d[0])
    
    starttest = time.time()
    right,wrong = test_discriminator_with_bleaching(d,acc_pos,px_test[0:100],py_test[0:100])
    endtest = time.time()
    print("time test = ",endtest - starttest)
    print("number of right result = ",right)
    print("number of wrong results = ",wrong)
    
    accuracy = ((right)/(right+wrong))*100
    print("accuracy by testing the model =",accuracy)
    
    
#    print(x_test.shape)
#    print(y_test.shape)
    
#    print(ty_test.shape)
#    ty_test.flatten()
#    print(ty_test)
#    print(tx_train[0].flatten().shape)
#    print(tx_train[0].flatten())