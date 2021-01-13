#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:28:36 2020

@author: caitlynjiang
"""
#use under python environment
#file name: prediction_probability.py

import tensorflow as tf
import argparse
import os
import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def get_parser():
    #discription of the py file
    parser = argparse.ArgumentParser(description="Output CNN model prediction for skin lesions classification")
    #add commands
    parser.add_argument('--csvname', '-n', default='untitled')
    parser.add_argument('--testdir', '-d', required=True) 
    parser.add_argument('--outputdir', '-o', default='') # default: the current directory
    
    return parser


def test_generator_init(test_dir):
    BATCH_SIZE=16    
    
    test_datagen=ImageDataGenerator(rescale=1./255)

    test_generator=test_datagen.flow_from_directory(
            test_dir,
            target_size=(224,224),
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False)
            # we test the images in the directories in sequence(shuffle=Flase)
            
    return test_generator


def prediction_probability_result(prediction,test_dir,output_dir,csv_name):
    
    class_names = ['melanoma', 'nevus', 'seborrheic_keratosis']
    rowname=class_names.copy()
    rowname.insert(0,'filename/probility')
    filenames=[]
    a = os.listdir(test_dir + 'test')
    #By defalt, the test directory that the user used should have a folder named 'test' under it
    a = [i for i in a if i != '.DS_Store']
    #Here, if you are using MACOS system, there will be an error because of the ".DStore" file.
    #The ".DStore" file need to be first deleted.
    a.sort()
    filenames += a
        
    step=len(class_names)

    #we build a matrix to store the prediction probability for each image
    final=int(len(prediction)/step)
    matrix=[[' ' for i in range(step+1)] for j in range(final+1)]
    #add the rownames
    matrix[0]=rowname
    #add the image filenames to the matrix
    row = 1
    for i in range(len(filenames)):
        matrix[row][0] = filenames[i]
        row += 1
        
    count=0
    for i in range(0,len(prediction),step):
        current_round=prediction[i:i+step]
        if count==final:
                break
        else:
            count += 1
        for j in range(1,step+1):
            matrix[count][j]=current_round[j-1]
            
    #here write the matrix into csv file   
    numpy.savetxt(output_dir+ csv_name + '.csv', matrix, fmt='%s', delimiter = ',')
    #a rough view for the result
    return matrix


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    csv_name = args.csvname
    test_dir = args.testdir + '/'
    output_dir = args.outputdir + '/'
    #excecute
    model=load_model('add3_3e_4.h5')
    #this is our model saved after training
    b = test_generator_init(test_dir)
    pred_keras = model.predict(b).ravel()
    prediction_probability_result(pred_keras,test_dir,output_dir,csv_name)

            
