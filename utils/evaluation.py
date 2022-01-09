import numpy as np
import os
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2

def plot(hist, plot_path, target, tl_alpha):
    
    # Save loss plot and dice score plot 

    train_loss = hist['final_loss']
    val_loss = hist['val_final_loss']
    acc = hist['final_dsc'] 
    val_acc = hist['val_final_dsc']
        
    epochs = np.arange(1, len(train_loss)+1,1)
    plt.plot(epochs,train_loss, 'b', label='Training Loss')
    plt.plot(epochs,val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Loss - {} segmentation (alpha {})'.format(target, tl_alpha))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plot_path, "loss_plot_{}_{}.png".format(target, tl_alpha)))

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training Dice Coefficient')
    plt.plot(epochs, val_acc, 'r', label='Validation Dice Coefficient')
    plt.grid(color='gray', linestyle='--')
    plt.legend()            
    plt.title('DSC  - {} segmentation (alpha {})'.format(target, tl_alpha))
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.savefig(os.path.join(plot_path, "dsc_plot_{}_{}.png".format(target, tl_alpha)))

def dice_score(y_pred, y_true):
    
    smooth = 1
    pred = np.ndarray.flatten(np.clip(y_pred,0,1))
    gt = np.ndarray.flatten(np.clip(y_true,0,1))
    intersection = np.sum(pred * gt) 
    union = np.sum(pred) + np.sum(gt)   
    
    return np.round((2 * intersection + smooth)/(union + smooth),decimals=5)

def performance_metrics(y_true, y_pred):
    
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    
    recall = (tp + smooth) / (tp + fn + smooth) # recall # Sensitivity 
    specificity = (tn + smooth) / (tn + fp + smooth) # Specificity 
    precision = (tp + smooth) / (tp + fp + smooth) # precision
    
    return [recall, specificity, precision]
    
def eval_model(model, X_test, Y_test, thresh=0.5):

    _, _, _, preds = model.predict(X_test)
    print("ground truth shape : ", X_test.shape) 
    print("prediction shape : ", preds.shape) 
    
    dice_score_list = np.zeros((len(X_test), 1))
    recall_list = np.zeros_like(dice_score_list)
    specificity_list = np.zeros_like(dice_score_list)
    precision_list = np.zeros_like(dice_score_list)
    
    for i in range(len(X_test)):
        
        gt_mask = Y_test[i:i+1]
        pred_mask = preds[i:i+1] 
        dice_score_list[i] = dice_score(pred_mask > thresh, gt_mask)
        
        rc, sp, prc = performance_metrics(gt_mask, pred_mask)
        recall_list[i] = rc
        specificity_list[i] = sp
        precision_list[i] = prc
        
    print('-'*30)
    print('USING HDF5 saved model at thresh=', str(thresh))
    
    print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dice_score_list)/100,  
        np.sum(recall_list)/100,
        np.sum(precision_list)/100))
    
    #plot precision-recall 
    y_true = Y_test.ravel() 
    y_preds = preds.ravel() 
    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
    plt.figure(20)
    plt.plot(recall,precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision") 
    plt.show() 

def plot_prediction_heart(X, Y, study_key_list, model, idx, save_fig_path,thresh=0.5):
    
    model = model
    input_ = X[idx:idx+1]
    mask_ = Y[idx:idx+1]
    study_key = study_key_list[idx]
    
    _, _, _, predicted_mask = model.predict(input_)
    
    predicted_mask = (predicted_mask > thresh).astype(np.uint8)
    
    plt.figure(figsize=(20,20))
    
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.axis('off')
    plt.imshow(input_[0], 'gray')
    
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    plt.imshow(mask_[0], 'gray')
    
    plt.subplot(1,3,3)
    plt.title("Predicted Mask for Heart")
    plt.axis('off')
    plt.imshow(predicted_mask[0][:, :, 0], 'gray')
    
    plt.savefig(os.path.join(save_fig_path, f"{study_key}.png"))
    

def plot_prediction_t4(X, Y, study_key_list, model, idx, save_fig_path, predict_thresh=0.5):
    
    model = model
    input_ = X[idx:idx+1]
    mask_ = Y[idx:idx+1]
    study_key = study_key_list[idx]

    _, _, _, predicted_mask = model.predict(input_)
    predicted_mask = (predicted_mask > predict_thresh).astype(np.uint8)
    
    result_mask = predicted_mask[0]
    thresh = cv2.threshold(result_mask, 0.5, 255, cv2.THRESH_BINARY)[1]
    
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    try:
        t4_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(t4_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        (a, b), (c, d), angle = rect
        # cv2.drawContours(ret, [box], -1, (0,255, 0), 2)
    except ValueError:
        print(study_key)
    
    plt.figure(figsize=(20,20))
    
    plt.subplot(1,3,1)
    plt.title("Input Image : {}".format(study_key))
    plt.axis('off')
    plt.imshow(input_[0], 'gray')
    
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    plt.imshow(mask_[0], 'gray')
    
    plt.subplot(1,3,3)
    plt.title("Predicted Mask for T4")
    plt.axis('off')
    plt.imshow(result_mask, 'gray')
    
    plt.show()
    
    plt.savefig(os.path.join(save_fig_path, f"{study_key}.png"))
    
def calculate_heart_area_length(X, study_key_list, model, idx, thresh=0.5):
    
    # Prediction 
    
    model = model
    input_ = X[idx:idx+1]
    study_key = study_key_list[idx]
    _, _, _, predicted_mask = model.predict(input_)
    predicted_mask = (predicted_mask > thresh).astype(np.uint8)
    heart_mask = predicted_mask[0][:, :, 0] # shape (512, 512) 
    
    ## heart area (A)
    A_pred = np.unique(heart_mask, return_counts=True)[1][1]
    
    ## heart height (L)
    
    # threshold
    thresh = cv2.threshold(heart_mask,0.5,255,cv2.THRESH_BINARY)[1]

    # get contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get area of largest contour -- assume it is heart   
    heart_contour = max(contours, key=cv2.contourArea)
    rect = cv2.boundingRect(heart_contour)
    
    x, y, w, h = rect
    L_pred = h
    
    return A_pred, L_pred
    
def calculate_t4_length(X, study_key_list, model, idx, predict_thresh=0.5):
    
    # Prediction 
    
    model = model
    input_ = X[idx:idx+1]
    study_key = study_key_list[idx]

    _, _, _, predicted_mask = model.predict(input_)
    predicted_mask = (predicted_mask > predict_thresh).astype(np.uint8)
    t4_mask = predicted_mask[0][:, :, 0]
    
    # threshold
    thresh = cv2.threshold(t4_mask,0.5,255,cv2.THRESH_BINARY)[1]

    # get contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get area of largest contour -- assume it is t4    
    # get min area rect of small contour (t4) 
    t4_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(t4_contour)
    
    (a, b), (c, d), angle = rect
    
    width = max(c, d)
    
    return np.round(width, 3)
    