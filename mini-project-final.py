#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:00:46 2020

bytlib load python-3.7.4
python3.7 mini-project-3.py

@author: zhaojingxian; coauthor: jianganlan
"""
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
import PIL
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


base_dir = '/public/workspace/3180111435bit/zjx/'
train_dir = base_dir + 'train/'
validation_dir= base_dir +'valid/'
test_dir= base_dir + 'test/'


subdirectories=os.listdir(test_dir)
subdirectories=[i for i in subdirectories if i != '.DS_Store']
#if your computer is mac, you have a DS.Store file, we need to excluded it
loc=0
length=[0,0,0]
for i in ['train/','valid/','test/']:
    c_dir=base_dir+i
    for j in subdirectories:
        length[loc]=length[loc]+len(os.listdir(c_dir+j))
        print(len(os.listdir(c_dir+j)))
    loc=loc+1
#length contains picture number for train,validation and test directory 


BATCH_SIZE=16
INPUT_SHAPE = (224,224,3)
TARGET_SIZE = (224,224)
classes = 3

NUM_EPOCHS = 100
INIT_LR = 1e-04

#data preprocessing：imagedatagenerator is used here to read data from directory, 
#data agumentation is performed to avoid overfitting
train_datagen=ImageDataGenerator(
    rescale=1./255,#value standardization
    rotation_range=40,#rotation for the image
    zoom_range=0.2,#bigger or smaller
    #randomly zoom the image between 0.8 and 1.2
    horizontal_flip=True
    #horizonal flip for the image
    )

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,#resize the picture to 224*224
    batch_size=BATCH_SIZE,#16 pictures a time to train the model
    class_mode='sparse',#we use 1D integer labels
    #melanoma:0 nevus:1 seborrheic_keratosis:2
    shuffle=True,
    )

validation_datagen=ImageDataGenerator(rescale=1./255)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False,
    )

test_datagen=ImageDataGenerator(rescale=1./255)


test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
    # we test the images in the directories in sequence(shuffle=Flase)
    )
#the training image number does not increase here, because generator is lazy load
#the code is wirtten with help of the example of Imagegenerator .flow_from_directory(directory) where
# you can use command+click to have a look



#check out generator
#print(train_generator.next()) you can see the detailed output for train generator for each batch 
for data_batch,labels_batch in train_generator:
    print('data batch shape:',data_batch.shape)#（16，224，224，3）
    print('label batch shape:',labels_batch.shape)#（16，）
    break



#develop the model
def model_construction():  
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',
                            input_shape=INPUT_SHAPE))#filer, kernel size 3*3
    model.add(layers.MaxPooling2D((2,2)))#down sample of feature map
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))#dropot layer to avoid overfitting
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(3,activation='softmax'))#multi-classification
    #check out the model structure
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=INIT_LR),
                  metrics=['acc'])
    return model
    

model=model_construction()


#fit the model with data
history=model.fit_generator(
    train_generator,
    steps_per_epoch= length[0] // BATCH_SIZE, #the number of batches in a epoch
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps= length[1] // BATCH_SIZE, #the number of batch used for validation
    )


#save a model
model.save('add3_3e_4.h5')
#if you want to reload the model we used
#from keras.models import load_model
#model=load_models('add3_3e_4.h5' )



#retrive the values for learning curve plotting
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

#get min(loss) and max(acc)
train_max_acc=max(acc)
validation_max_acc=max(val_acc)
validation_min_loss=min(val_loss)
train_min_loss=min(loss)

#plot accuracy curve for validation and training
def accuracy_curve_plot(epochs_number,train_accuracy,validation_accuracy):
    plt.plot(epochs_number,train_accuracy,label='Training accuracy')
    plt.plot(epochs_number,validation_accuracy,label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.savefig("accuracy_curve.png")

accuracy_curve_plot(epochs,acc,val_acc)

#plot loss curve for traning and validation
def loss_curve_plot(epochs_number,train_loss,validation_loss):
    plt.plot(epochs,train_loss,label='Training loss')
    plt.plot(epochs,validation_loss,label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    plt.savefig("loss_curve.png")
    
loss_curve_plot(epochs,loss,val_loss)



#evaluation of model
test_loss,test_acc = model.evaluate_generator(test_generator, steps= length[2] // BATCH_SIZE +1)
print('Loss= ', test_loss ,'Acurracy= ', test_acc)



#A function to write predicted probabilities into a csv file
class_names = ['melanoma', 'nevus', 'seborrheic_keratosis']
def prediction_probability_result(prediction,test_dir,csvname):
    rowname=class_names.copy()
    rowname.insert(0,'filename/probility')
    filenames=[]
    for i in class_names: 
        print(i)
        a = os.listdir(test_dir+i)
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
    np.savetxt(csvname+'.csv', matrix, fmt='%s', delimiter = ',')
    #a rough view for the result
    return matrix 
 

#figure prediction:prediction probability for each picture
pred_keras = model.predict(test_generator).ravel()
prediction_probability_result(pred_keras,test_dir,'prediction_probability')



#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True category')
    plt.xlabel('Predicted category')


#obtain predicted probabilities for confusion matrix
scores = model.predict_generator(
    generator=test_generator, 
    steps=length[2] // BATCH_SIZE + 1)
#retrieve the largest probability index as predicted result
#combine with true catogory from test_generator to get cnf_matrix array
cnf_matrix = confusion_matrix(test_generator.classes, list(map(lambda x: np.argmax(x), scores)))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')



#ROC curve and AUC score
def plot_roc_auc(y_true, y_pred):

    # initialize dictionaries and array
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(3)
    
    # prepare for figure
    plt.figure()
    colors = ['aqua', 'cornflowerblue']

    # for both classification tasks (categories 1 and 2)
    for i in range(2):
        # obtain ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        # obtain ROC AUC
        roc_auc[i] = auc(fpr[i], tpr[i])
        # plot ROC curve
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve for task {d} (area = {f:.3f})'.format(d=i+1, f=roc_auc[i]))
    # get score for category 3
    mean_socre = np.average(roc_auc[:2])
    
    # format figure
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()
    
    # print scores
    for i in range(2):
        print('Task {d} Score: {f:.3f}'. format(d=i+1, f=roc_auc[i]))
    print('Mean Score:', round(mean_socre,3))
    
    
# get ground truth labels for test dataset
    #- Task 1: Malignant vs non-malignant.
    #- Task 2: Keratinocytic vs melanocytic.
truth = pd.read_csv('/your/dir/ground_truth.csv')
y_true = truth.as_matrix(columns=["task_1", "task_2"])

# get model predictions for test dataset
y_pred = pd.read_csv('/your/dir/prediction_probability.csv')
y_pred = y_pred.as_matrix(columns=["melanoma", "seborrheic_keratosis"])

# plot ROC curves and print scores
plot_roc_auc(y_true, y_pred)




                    
   
