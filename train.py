"""
A Deep Learning Method to Detect Dogs' Heart Disease from Simple X-Rays: Applying an improved attention U-Net for size-unbalanced regions of interest
"""

## Load Libraries and Modules 

import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import keras 
from albumentations import * 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import cv2 
import time
import h5py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import glorot_normal, random_normal, random_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.models import load_model

from model.losses import *
from model.backbone import *
from utils.image_module import *

import matplotlib.pyplot as plt 
plt.rc('axes', unicode_minus=False)
from sklearn.model_selection import train_test_split

sys.path.append("..")

## Arguments 

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Improved Attention U-Net Training')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--seed', default=2021, help="seed for numpy and tensorflow")
parser.add_argument('--batch-size', default=16, type=int, help="Size of mini batch (default 16)")
parser.add_argument('--image-size', default=512, type=int, help="Size of input image (default 512 x 512)")
parser.add_argument('--epochnum', default=100, type=int, help="Number of total epochs to run")
parser.add_argument('--alpha', default=0.5, help="Hyperparameter alpha for tversky loss function. 0.7 is recommended for heart segmentation, and 0.2 for T4 body segmentation")
parser.add_argument('--learning-rate', default=0.001, help="Initial learning rate (default 0.001)")
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--reduce-lr-factor', default=0.1, type=float, help="Weight decay factor for reduceLROnPlateau callback")
parser.add_argument('--reduce-lr-patience', default=10, type=int, help="Patience for reduceLROnPlateau callback")
parser.add_argument('--early-stop-patience', default=15, type=int, help="Patience for early stopping callback")
parser.add_argument('--gpu', default=3, type=int, help="GPU id to use (default = 3)")
parser.add_argument('--target', default="HEART", type=str, help="Target organ for semantic segmentation (default=HEART)")
parser.add_argument('--test-size', default=100, type=int, help="Size of test set size to split from dataset")
parser.add_argument('--weight-path', type=str, help="Path to store weight (hdf5 file)")
parser.add_argument('--plot-path', type=str, help="Path to store loss and dsc plot (png file)")
parser.add_argument('--test', default=True, help="Infer test results (boolean flag, default = True)", type=str2bool)
parser.add_argument('--test-plot', default=False, help="Save test figure (input image & ground truth & predicted mask) (boolean flag, default = False)", type=str2bool)
parser.add_argument('--test-plot-path', type=str, help="Path to store test plot (png file)")

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

## Set GPU settings

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
gpu_num = args.gpu 
os.environ["CUDA_VISIBLE_DEVICES"]=",".join([i+1 for i in range(gpu_num)])
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2' 

mirrored_strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

## Seed setting

seed = args.seed

np.random.seed(seed)
tf.random.set_seed(seed)

## Hyperparaemter setting

MODEL = "attention_unet"
TARGET = args.target # Target organ 
tl_alpha = args.alpha # This is alpha used for tversky loss function 

IMAGE_SIZE = args.image_size
IMG_CHANNELS = 1
EPOCHNUM = args.epochnum
BATCH_SIZE = args.batch_size
input_size = (IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)

def main():
    
    args = parser.parse_args()
    
    # Data Loading
    
    if TARGET.upper() == "HEART" : # Heart segmentation
    
        X_all, Y_all, study_key_list = load_heart_data(args.data, IMAGE_SIZE)
        
    else : # T4 segmentation
        TARGET = "T4"
        X_all, Y_all, study_key_list = load_t4_data(args.data, IMAGE_SIZE)
        
    # Split data 
    X_train, X_test, Y_train, Y_test, sk_train, sk_test = train_test_split(X_all, Y_all, study_key_list, test_size=args.test_size, random_state=seed)
    X_train, X_valid, Y_train, Y_valid, sk_train, sk_valid = train_test_split(X_train, Y_train, sk_train, test_size=args.test_size, random_state=seed)

    print("============== Original Dataset (Before Augmentation) ==============")
    print("Training data shape : ", X_train.shape)
    print("Validation data shape : ", X_valid.shape)
    print("Test data shape : ", X_test.shape)

    print("Training label shape : "Y_train.shape)
    print("Validation label shape: "Y_valid.shape)
    print("Test label shape : "Y_test.shape)

    # Data Augmentation
    # You can change or add augmentation strategy by fixing augment_data function in image_module.py (in utils directory)

    X_train_augmented, Y_train_augmented = augment_data(X_train, Y_train, IMAGE_SIZE)

    print("============== Augmented Training Set ==============")
    print("Training data (augmented) shape : ", X_train_augmented.shape)
    print("Training label (augmented) shape : ", Y_train_augmented.shape)

    weight_path = args.weight_path

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # Defining Model

    optimizer = Adam(learning_rate=args.learning_rate)
    lossfxn = ftl_wrapper(tl_alpha)
    final_loss = tl_wrapper(tl_alpha)
    metric = dsc

    loss = {'pred1':lossfxn, # focal tversky loss (FTL)
            'pred2':lossfxn, # focal tversky loss (FTL)
            'pred3':lossfxn, # focal tversky loss (FTL)
            'final': final_loss} # Tversky loss (TL) 

    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}

    # Training
    ## Load improved attention u-net 
    with mirrored_strategy.scope():
        model = attn_reg(input_size)
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=[dsc])

    ## Callbacks

    check_point = ModelCheckpoint(
        weight_path+"weights_{epoch:02d}_{loss:.4f}.hdf5", 
        monitor='val_final_dsc', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max')

    reduceLROnPlat = ReduceLROnPlateau(
          monitor="val_loss",
          factor=args.reduce_lr_factor,
          patience=args.reduce_lr_patience,
          verbose=1,
          mode="min",
          min_delta=0.0001,
          cooldown=2,
          min_lr=1e-6)

    early = EarlyStopping(monitor="val_dsc", mode="max", patience=args.early_stop_patience)

    callbacks_list = [check_point, early, reduceLROnPlat]

    ## Input Pyramid 

    gt1_train_augmented = Y_train_augmented[:,::8,::8,:]
    gt2_train_augmented = Y_train_augmented[:,::4,::4,:]
    gt3_train_augmented = Y_train_augmented[:,::2,::2,:]
    gt4_train_augmented = Y_train_augmented

    gt_train_augmented = [gt1_train_augmented, gt2_train_augmented, gt3_train_augmented, gt4_train_augmented]

    gt1_train = Y_train[:,::8,::8,:]
    gt2_train = Y_train[:,::4,::4,:]
    gt3_train = Y_train[:,::2,::2,:]
    gt4_train = Y_train

    gt_train = [gt1_train, gt2_train, gt3_train, gt4_train]

    gt1_valid = Y_valid[:,::8,::8,:]
    gt2_valid = Y_valid[:,::4,::4,:]
    gt3_valid = Y_valid[:,::2,::2,:]
    gt4_valid = Y_valid

    gt_valid = [gt1_valid,gt2_valid,gt3_valid,gt4_valid]

    ## Model Training 

    hist = model.fit(X_train_augmented, gt_train_augmented, epochs=EPOCHNUM, batch_size=BATCH_SIZE, 
                validation_data=(X_valid, gt_valid), 
                shuffle = True, verbose=1, callbacks=callbacks_list)
    
    ## Save loss plot and dice score plot 
    # You can change plot name format by fixing plot function in image_module.py (in utils directory)
    
    plot(hist, args.plot_path, TARGET, tl_alpha)
    
    if args.test : 
        
        eval_model(model, X_test, Y_test) 
        
    if args.test_plot :
        
        if TARGET.upper() == "HEART" :
            
            for i in range(len(sk_test)):
                plot_prediction_heart(X_test, Y_test, sk_test, model, i, args.test_plot_path)
                
        else : # T4 
            
            for i in range(len(sk_test)):
                plot_prediction_t4(X_test, Y_test, sk_test, model, i, args.test_plot_path)