#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:28:00 2020

@author: zhaojingxian
"""

#no need to peform in usual cases
#see discussion part 3 in the report

import os
import matplotlib.pyplot as plt
img=plt.imread('/Users/zhaojingxian/data/train/nevus_old/ISIC_0012464.jpg')
img.shape()

len(os.listdir('/Users/zhaojingxian/data/train/melanoma'))
len(os.listdir('/Users/zhaojingxian/data/train/nevus_old'))#1372
len(os.listdir('/Users/zhaojingxian/data/train/seborrheic_keratosis'))#254
len(os.listdir('/Users/zhaojingxian/data/train/nevus'))#274
#extract a portion of images from nevus_old
import os, random, shutil
def copyFile(fileDir):
        pathDir = os.listdir(fileDir)   
        filenumber=len(pathDir)
        rate=0.2    # extract 20%
        picknumber=int(filenumber*rate) 
        sample = random.sample(pathDir, picknumber)  
        print (sample)
        for name in sample:
                shutil.copy(fileDir+name, tarDir+name)
        return
  
    
base_dir = '/Users/zhaojingxian/data/'
train_dir = base_dir + 'train/'
fileDir=train_dir+'nevus_old/' 
tarDir = '/Users/zhaojingxian/data/train/nevus'
copyFile(fileDir)
