# Original creator: Rajshree Daulatabad
# Modified by: Parinita Edke
# Date of first modification: 21st February, 2021

import torch
import random
import numpy as np
import pandas as pd
import time
from torch.autograd import Variable
from torchvision import transforms
import pickle
import torch.nn as nn
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import glob
    
# Parinita's imports
import pathlib
import segmentation_models_pytorch as smp
import albumentations as albu

random.seed(30)

######## START: Parinita's edits ########

# Use CPU
DEVICE = torch.device("cpu")

# Inserting Mauro's segmentation code here
def to_tensor(image, **kwargs):
    """
       Function to change image channels to Pytorch format

       image: Image to be formatted
    """
    return image.transpose(2, 0, 1).astype('float32')

def segment(input):
    """
       Function for segmenting an image

       input: Image (numpy array) to be segmented
    """

    # Create model
    model = smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=1, activation='sigmoid')

    # Loading weights
    model.load_state_dict(torch.load('/home/midata_gpu/Projects/Weights/Effusion_Segmentation/weights_knees_segmentation.pth',\
          map_location=DEVICE)['model_state_dict'])

    # Prepare the model
    model.to(DEVICE)
    model.eval()

    # Preprocessing - Transformations
    SEG_TRANSFORMS = albu.Compose([
        albu.Resize(256, 256),
        albu.Lambda(image=smp.encoders.get_preprocessing_fn('efficientnet-b4','imagenet')),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ])

    input_tensor = torch.tensor(SEG_TRANSFORMS(image=input)['image'])

    # Get Segmentation
    with torch.no_grad():
        mask = model(input_tensor.unsqueeze(0))
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        mask = mask.squeeze().numpy() * 255

    # Post Processing - Upscaling
    resized = cv2.resize(np.uint8(mask), (input.shape[1], input.shape[0]), interpolation = cv2.INTER_AREA)

    return resized

######## END: Parinita's edits ########

"""
Function name : test_one_patch
Input Parameters : 
                    model - saved cnn model object reference
                    device - device reference to run the model on gpu 
                    input_patch_img - the 64x64 patch input image to identify the patterns   
                    transform - pytorch's transform object to be applied on the input image to prepare it for the cnn input
Output            : pred - CNN prediction in form of 5 probabilities for each pattern types
"""

def test_one_patch(model, device, input_patch_img, transform):
    #print(type(input_patch_img))
    img = np.zeros((input_patch_img.shape[0],input_patch_img.shape[1], 3) )
    img[:,:,0] = input_patch_img
    img[:,:,1] = input_patch_img
    img[:,:,2] = input_patch_img
    input_patch_img = img
    model.eval()
    with torch.no_grad():
        images = transform(input_patch_img)
        images = torch.unsqueeze(images, 0)
        images = Variable(images)
        x = images.to(device)
        output = model(x)
        pred = torch.sigmoid(output).data.cpu().numpy().tolist()[0]
    return pred

test_transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.ToTensor(),
    transforms.Normalize([0.1466, 0.1465, 0.1462], [0.1542, 0.1530, 0.1505])
])

"""
Function name    : get_64x64_patches_arnd_effusion
Description      : gets all the 64x64 mutually exclusive patches from effusion area only
Input Parameters : single channel inout ultrasound image and the corresponding single channel effusion mask image 
Output           : list of [x,y] coordinates representing left top cornors of the 64x64 patches
"""

def get_64x64_patches_arnd_effusion(input_us_img, input_mask):
    patches=[]
    x, y, w, h = cv2.boundingRect(input_mask)
 #   print(x, y, w, h)
    if w < 64 :
        w_new=64
    else:
        w_new=w
    if h < 64 :
        h_new=64
    else:
        h_new=h
    for i in range(int(w_new/64)):
        for j in range(int(h_new/64)):
            if np.sum(input_mask[y+j*64:y+j*64+64, x+i*64:x+i*64+64])>0:
                patches.append([x+i*64, y+j*64]) # add the patch only if the corresponding mask area overlaps with effusion area
#    print(patches)
    return patches
"""
Function name    : get_svm_pred
Description      : Provides the svm prediction results (for blood detection) based on the 5 pattern labels provided 
Input parameters : list of 5 labels, for example input_X = [6,0,5,0,0]
Output           : svm prediction results [0] or [1]
"""
def get_svm_pred(input_X):
    model_filename = 'weights_blood_classification_svm.sav'
    saved_svm_model = pickle.load(open(model_filename, 'rb'))
#    print(input_X)
    # print(saved_svm_model.predict_proba[np.array(input_X)])
    return saved_svm_model.predict_proba([np.array(input_X)])

"""
************************** MAIN FUNCTION TO CALL ******************************
Predicts the blood detection by identifying the number of different patterns present in the effusion 
and then calling the blood classifier (svm based)
Steps :
        1) 64x64 patch images are extracted from the effusion area of input ultrasound image 
        2) Each of the patch images are passed through the trained CNN (VGG19 based) which provides predictions on presence of 5 patterns
        3) The cutoffs for each pattern labels provided as input (default = 0.5) is applied on the probabilities
        4) Total sum count is calculated to find out how many patches belong to each category (fully granular, partial granular, debris, tissue, pure effusion)
        5) The counts are given to SVM model to predict the blood detection

Function Name    : predict_blood_detection
Input parameters : 
            input_us_img - single channel input ultrasound image
            input_mask   - single channel binary mask
            cutoffs for each pattern labels to be considered for calculation of counts for svm
            cutoff for final blood detection (this parameter is not used as of now because the classifier is SVM which does not output probability, this paramter is for future in case we use other classifiers)
*******************************************************************************
"""
def predict_blood_detection(input_us_img, input_mask, label1_cutoff=0.5, label2_cutoff=0.5, label3_cutoff=0.5, label4_cutoff=0.5, label5_cutoff=0.5, blood_det_cutoff=0.5):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False)
    model.classifier[6] = nn.Linear(4096,5)
    device = torch.device("cpu")
    model.to(device)
    learning_rate = 0.000001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #model = torch.load("./weights_blood_classification_pattern.mdl", map_location=torch.device('cpu')) #../../../../models/pattern_effnet-v1_Epoch"+str(epoch)+".mdl")
    model = torch.load("./pari_weights_blood_classification_pattern.mdl", map_location=torch.device('cpu'))

    #Trying something to see if I can get rid of the SourceChangeWarning
    #torch.save(model, "./pari_weights_blood_classification_pattern.mdl")
    #print("Saved the model")

    patches_xy = get_64x64_patches_arnd_effusion(input_us_img, input_mask)
    count_label1 = 0
    count_label2 = 0
    count_label3 = 0
    count_label4 = 0
    count_label5 = 0

    print("	Length of patches array:", len(patches_xy))

    for patch_xy in patches_xy:
        pred_one = test_one_patch(model, device, input_us_img[patch_xy[1]:patch_xy[1]+64, patch_xy[0]:patch_xy[0]+64], test_transform)
        count_label1 += (int(pred_one[0] >= label1_cutoff))
        count_label2 += (int(pred_one[1] >= label2_cutoff))
        count_label3 += (int(pred_one[2] >= label3_cutoff))
        count_label4 += (int(pred_one[3] >= label4_cutoff))
        count_label5 += (int(pred_one[4] >= label5_cutoff))
    
    
    probs = get_svm_pred([count_label1, count_label2, count_label3, count_label4, count_label5])[0]
    # Get results
    prob_pos = probs[1].item()
    predicted = int(prob_pos > 0.5)

    return prob_pos, predicted


"""
Function Name    : manual_64x64crops_predict_blood_detection
Input parameters : 
            input_us_img_patch_array - array of single channel input US crops
            cutoffs for each pattern labels to be considered for calculation of counts for svm
            cutoff for final blood detection (this parameter is not used as of now because the classifier is SVM which does not output probability, this paramter is for future in case we use other classifiers)

"""
# This is a modified version of Rajshree's original function. This is meant to be used when we have manual crops and don't need to segment the original US image
def manual_64x64crops_predict_blood_detection(input_us_img_patch_array, label1_cutoff=0.5, label2_cutoff=0.5, label3_cutoff=0.5, label4_cutoff=0.5, label5_cutoff=0.5, blood_det_cutoff=0.5):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=False)
    model.classifier[6] = nn.Linear(4096,5)
    device = torch.device("cpu")
    model.to(device)
    learning_rate = 0.000001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #model = torch.load("./weights_blood_classification_pattern.mdl", map_location=torch.device('cpu')) #../../../../models/pattern_effnet-v1_Epoch"+str(epoch)+".mdl")
    model = torch.load("./pari_weights_blood_classification_pattern.mdl", map_location=torch.device('cpu'))

    #Trying something to see if I can get rid of the SourceChangeWarning
    #torch.save(model, "./pari_weights_blood_classification_pattern.mdl")
    #print("Saved the model")

    count_label1 = 0
    count_label2 = 0
    count_label3 = 0
    count_label4 = 0
    count_label5 = 0

    for crop in input_us_img_patch_array:
        #print(crop)
        #if (type(crop) is not None):
        #    print("Reached here")
        pred_one = test_one_patch(model, device, crop, test_transform)
        count_label1 += (int(pred_one[0] >= label1_cutoff))
        count_label2 += (int(pred_one[1] >= label2_cutoff))
        count_label3 += (int(pred_one[2] >= label3_cutoff))
        count_label4 += (int(pred_one[3] >= label4_cutoff))
        count_label5 += (int(pred_one[4] >= label5_cutoff))
    
    
    probs = get_svm_pred([count_label1, count_label2, count_label3, count_label4, count_label5])[0]
    # Get results
    prob_pos = probs[1].item()
    predicted = int(prob_pos > 0.5)

    return prob_pos, predicted



###################################################################################################################################

# PARINITA'S CODE TO GET PREDICTIONS ON THE TEST DATASET
print("Generating blood yes/no predictions using Rajshree's model...")

real_blood_dataset = pathlib.Path('../RB_split/Split_dataset/test/Real_Blood/').glob('*')
real_effusion_dataset = pathlib.Path('../RE_split/Split_dataset/test/Real_Effusion').glob('*')
output_file = 'testing_real_data_blood_class_results.csv'

r = []
num_correct = 0
i = 1
TN, TP, FN, FP = 0, 0, 0, 0

print("Iterating through real blood image folder...")
for path in real_blood_dataset:
    image_path = str(path)

    # retrieve file name
    f_name = (str(path).split('/')[-1]).rstrip()
    print("Image: ", f_name)

    # Read image
    print(" Reading image...")
    image = cv2.imread(image_path)
    grey_scale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Segment the image
    print(" Segmenting...")
    segmentation = segment(image)

    # Call prediction function
    print(" Predicting presence of blood...")
    prob_pos, predicted = predict_blood_detection(grey_scale_image, segmentation)

    # Tally correct predictions to get accuracy
    if predicted == 1:
        num_correct += 1
        TP += 1
    elif predicted == 0:
        FN += 1 

    # Save to results array
    print(" Saving results to array...")
    print(" ID: ", 'RBlood_' + f_name)
    r.append({'IDs': 'RBlood_' + f_name, 'Level': 1 ,'Blood Prob': prob_pos, 'Blood Pred': predicted})


print("Finished iterating through real blood image folder.")
print("Length of r array: ", len(r))

print("Iterating through real effusion image folder...")
for effusion_path in real_effusion_dataset:
    image_path = str(effusion_path)

    # retrieve file name
    f_name = (str(effusion_path).split('/')[-1]).rstrip()
    print("Image: ", f_name)

    # Read image
    print(" Reading image...")
    image = cv2.imread(image_path)
    grey_scale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Segment the image
    print(" Segmenting...")
    segmentation = segment(image)

    # Call prediction function
    print(" Predicting presence of blood...")
    prob_pos, predicted = predict_blood_detection(grey_scale_image, segmentation)

    # Tally correct predictions to get accuracy
    if predicted == 0:
        num_correct += 1
        TN += 1
    elif predicted == 1:
        FP += 1

    # Save to results array
    print(" Saving results to array...")
    print(" ID: ", 'REffusion_' + f_name)
    r.append({'IDs': 'REffusion_' + f_name, 'Level': 0 ,'Blood Prob': prob_pos, 'Blood Pred': predicted})

print("Finished iterating through real effusion image folder.")


###################################################################################################################################

print("Length of r array: ", len(r))


results = pd.DataFrame(r, columns=['IDs', 'Level', 'Blood Prob', 'Blood Pred'])
#results.to_csv('synthetic_data_blood_class_results.csv')
results.to_csv(output_file)
print("Accuracy of predictions for real data test dataset: ", num_correct/len(r))
print("TN: {}  TP: {}  FN: {}  FP: {}".format(TN, TP, FN, FP))
print("Done!")

