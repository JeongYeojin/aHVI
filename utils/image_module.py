from tqdm import tqdm
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from albumentations import * 

from skimage.measure import label
from skimage import morphology
from PIL import Image
from random import sample 

def center_crop_clahe(img_path, cropped_img_path, clahe_img_path): 
    
    # Center crop
    
    img = Image.open(img_path)
    width, height = img.size
    new_width = min(width, height)
    new_height = new_width
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    img_cropped = img.crop((left, top, right, bottom))
    
    img_cropped.save(cropped_img_path)
    
    # CLAHE 
    
    img_cropped = cv2.imread(cropped_img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_cropped_clahe = clahe.apply(img_cropped)
    
    cv2.imwrite(clahe_img_path, img_cropped_clahe)

def json_to_label(study_key):

    # labelme conda environment must be activated
    # Ouput folder -> img.png (input image), label.png (annotation image file), label_names.txt, label_viz.png (annotation stacked on original image)
    os.system("labelme_json_to_dataset {}_cropped_clahe.json -o {}_json".format(study_key, study_key))

def label_to_heart_mask(base_dir, target_dir, IMAGE_SIZE):
    
    # Make Heart only binary mask from label.png (generated from labelme, has both mask (heart & t4)) 
    # base_dir : ex) aHVI/data
    # target_dir : ex) aHVI/data/mask_heart
    
    label_dir = os.path.join(base_dir, "mask") # aHVI/data/mask
    label_list = sorted(glob.glob(os.path.join(label_dir, "*.png")))
    
    study_key_list = [x.split("/")[-1] for x in label_list]
    study_key_list = [x.split("_")[0] for x in study_key_list]
    
    for idx, file_path in tqdm(enumerate(label_list)):
        y = Image.open(file_path)
        y = asarray(y)
        y = cv2.resize(y, (IMAGE_SIZE, IMAGE_SIZE))
        y = y.astype(np.float32)
        y = np.where(y == 1, 1, 0)
        y = morphology.remove_small_objects(label(y), 100) # Remove small borderlines 
        y = np.where(y != 0, 1, 0)
        im = Image.fromarray((y * 255).astype(np.uint8))
        im.save(os.path.join(target_dir, "{}.jpg".format(study_key_list[idx])))
        
def load_heart_data(base_dir, IMAGE_SIZE):
    
    # base_dir : path which contains input and mask directory (ex. aHVI/data)
    # Ouput : X_all, Y_all, study_key_list (study key list (sorted)) 
    
    input_dir = os.path.join(base_dir, "input") # ex) aHVI/data/input/
    mask_dir = os.path.join(base_dir, "mask_heart") # ex) aHVI/data/mask_heart/
    
    input_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    study_key_list = [x.split("/")[-1] for x in input_list]
    study_key_list = [x.split(".")[0] for x in study_key_list]
    N = len(study_key_list) # number of images 
    
    X_all = []
    Y_all = []
    
    for _, file_path in tqdm(enumerate(input_list)):
        x = cv2.resize(cv2.imread(file_path), (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0 # Normalize 
        X_all.append(x.astype(np.float32)[:, :, 0]) 

    for x in tqdm(X_all):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    for _, file_path in tqdm(enumerate(mask_list)):
        y = Image.open(file_path)
        y = asarray(y)
        y = cv2.resize(y, (IMAGE_SIZE, IMAGE_SIZE))
        y = y.astype(np.float32)
        y = y / 255.0 # Normalize
        y = np.where(y > 0.5, 1, 0)
    
        Y_all.append(y)
    
    X_all = np.array(X_all)
    X_all = X_all.reshape(N, IMAGE_SIZE, IMAGE_SIZE, 1) # (N, IMAGE_SIZE, IMAGE_SIZE, 1)
    X_all = X_all.astype(np.float32)
    
    Y_all = np.array(Y_all)
    Y_all = Y_all.reshape(N, IMAGE_SIZE, IMAGE_SIZE, 1) # (N, IMAGE_SIZE, IMAGE_SIZE, 1)
    Y_all = Y_all.astype(np.float32)
    
    print("X shape : ", X_all.shape)
    print("X dtype : ", X_all.dtype)
    print("Y shape : ", Y_all.shape)
    print("Y dtype : ", Y_all.dtype)
    
    return X_all, Y_all, study_key_list 

def label_to_t4_mask(base_dir, target_dir, IMAGE_SIZE):
    
    # Make Heart only binary mask from label.png (generated from labelme, has both mask (heart & t4)) 
    # base_dir : aHVI/data
    # target_dir : aHVI/data/mask_t4
    
    label_dir = os.path.join(base_dir, "mask") # aHVI/data/mask
    label_list = sorted(glob.glob(os.path.join(label_dir, "*.png")))
    
    study_key_list = [x.split("/")[-1] for x in label_list]
    study_key_list = [x.split("_")[0] for x in study_key_list]
    
    for idx, file_path in tqdm(enumerate(label_list)):
        y = Image.open(file_path)
        y = asarray(y)
        y = cv2.resize(y, (IMAGE_SIZE, IMAGE_SIZE))
        y = y.astype(np.float32)
        y = np.where(y == 2, 1, 0)
 
        im = Image.fromarray((y * 255).astype(np.uint8))
        im.save(os.path.join(target_dir, "{}.jpg".format(study_key_list[idx])))


def load_t4_data(base_dir, IMAGE_SIZE):
    
    # base_dir : path which contains input and mask directory (ex. aHVI/data)
    # Ouput : X_all, Y_all, study_key_list (study key list (sorted)) 
    
    input_dir = os.path.join(base_dir, "input") # ex) aHVI/data/input/
    mask_dir = os.path.join(base_dir, "mask_t4") # ex) aHVI/data/mask_t4/
    
    input_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    study_key_list = [x.split("/")[-1] for x in input_list]
    study_key_list = [x.split(".")[0] for x in study_key_list]
    N = len(study_key_list) # number of images 
    
    X_all = []
    Y_all = []
    
    for _, file_path in tqdm(enumerate(input_list)):
        x = cv2.resize(cv2.imread(file_path), (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0 # Normalize 
        X_all.append(x.astype(np.float32)[:, :, 0]) 

    for x in tqdm(X_all):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
    for _, file_path in tqdm(enumerate(mask_list)):
        y = Image.open(file_path)
        y = asarray(y)
        y = cv2.resize(y, (IMAGE_SIZE, IMAGE_SIZE))
        y = y.astype(np.float32)
        y = y / 255.0 # Normalize 
        y = np.where(y > 0.5, 1, 0)
        
        Y_all.append(y)
        
    X_all = np.array(X_all)
    X_all = X_all.reshape(N, IMAGE_SIZE, IMAGE_SIZE, 1) # (N, IMAGE_SIZE, IMAGE_SIZE, 1)
    X_all = X_all.astype(np.float32)
    
    Y_all = np.array(Y_all)
    Y_all = Y_all.reshape(N, IMAGE_SIZE, IMAGE_SIZE, 1) # (N, IMAGE_SIZE, IMAGE_SIZE, 1)
    Y_all = Y_all.astype(np.float32)
    
    print("X shape : ", X_all.shape)
    print("X dtype : ", X_all.dtype)
    print("Y shape : ", Y_all.shape)
    print("Y dtype : ", Y_all.dtype)
    
    return X_all, Y_all, study_key_list 
    

def plotMask(X, Y):
        
    # For Sanity Check
    # Plot 6 images & mask 
    
    sample_idx = sample(range(len(X)), 6) 
    img = []
    
    for i in sample_idx:
        left = X[i][:, :, 0] # (IMAGE_SIZE, IMAGE_SIZE)
        right = Y[i][:, :, 0] # (IMAGE_SIZE, IMAGE_SIZE)
        combined = np.hstack((left,right))
        img.append(combined)
        
    for i in range(0,6,3):

        plt.figure(figsize=(25,10))
        
        plt.subplot(2,3,1+i)
        plt.imshow(img[i], cmap=plt.cm.bone)
        
        plt.subplot(2,3,2+i)
        plt.imshow(img[i+1], cmap=plt.cm.bone)
        
        plt.subplot(2,3,3+i)
        plt.imshow(img[i+2], cmap=plt.cm.bone)
        
        plt.show()
        
        
def augment_data(X_original, Y_original, IMAGE_SIZE):
    
    X_augmented = X_original.copy()
    Y_augmented = Y_original.copy()
    
    for x, y in tqdm(zip(X_original, Y_original), total=len(X_original)):
        
        aug = ElasticTransform(p=1.0)
        augmented = aug(image=x, mask=y)
        x1 = augmented["image"]
        y1 = augmented["mask"]
        
        X_augmented = np.append(X_augmented, x1.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)
        Y_augmented = np.append(Y_augmented, y1.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)      
        
        aug = RandomBrightnessContrast(p=1.0)
        augmented = aug(image=x, mask=y)
        x3 = augmented["image"]
        y3 = augmented["mask"]
        
        X_augmented = np.append(X_augmented, x3.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)
        Y_augmented = np.append(Y_augmented, y3.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)      
        
        aug = GaussianBlur(p=1.0)
        augmented = aug(image=x, mask=y)
        x3 = augmented["image"]
        y3 = augmented["mask"]
        
        X_augmented = np.append(X_augmented, x3.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)
        Y_augmented = np.append(Y_augmented, y3.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1), axis=0)      
            
    return X_augmented, Y_augmented
        

    
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
    
    
    
    
    