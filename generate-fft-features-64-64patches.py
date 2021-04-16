# Main Author: Rajshree Daulatabad
# Modified by: Parinita Edke
# Date of creation: 22nd January 2021

import os
import cv2
import numpy as np
from math import sqrt
import pandas as pd

# folder_path = "./patches_sample/"
synthetic_data_path = "./patches_sample/"
train_real_blood_path = "./train_real_blood_patches/"
train_real_effusion_path = "./train_real_effusion_patches/"

test_real_blood_path = "./test_real_blood_patches/"
test_real_effusion_path = "./test_real_effusion_patches/"

# datasets = [synthetic_data_path, train_real_blood_path, train_real_effusion_path]
datasets = [test_real_blood_path, test_real_effusion_path]

fft_features=[]
image_labels = []

nonetype_images = []
print("Generating fft features for images")

for dataset in datasets:
    images_path = os.listdir(dataset)
    # print('Images_path:', images_path)
    # print("Dataset: ", dataset)
    for n, image in enumerate(images_path):
        # print('	File: ', image)
        subfolder_path = os.listdir(dataset+ image)
        for m, subfolder in enumerate(subfolder_path):
            # print('		Subfolder:', subfolder)

            img = cv2.imread(dataset+image+'/'+subfolder,cv2.IMREAD_GRAYSCALE)
            img_float32 = np.float32(img)/255.0

            dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            shiftedFFTMagnitude = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

            if img is None:
                nonetype_images.append(image+'/'+subfolder)

            else:
                rows, cols = img.shape
                midRow = rows/2+1
                midCol = cols/2+1
                maxRadius = int(sqrt((np.max([rows,cols]))**2 + (np.max([rows,cols]))**2))
                # print("Max radius: ", maxRadius, "Rows: ", rows, "Cols: ", cols)

                radialProfile = np.zeros((maxRadius, 1))
                count = np.zeros((maxRadius, 1))

                for col in range(cols):
                    for row in range(rows):
                        radius = sqrt((row - midRow)**2 + (col - midCol)**2);
                        thisIndex = int(radius) ;
                        radialProfile[thisIndex] = radialProfile[thisIndex] + shiftedFFTMagnitude[row, col];
                        count[thisIndex] = count[thisIndex] + 1;
                        # print(col, row, thisIndex, radialProfile[thisIndex], count[thisIndex])
      
                # Get average
                avg=[]
                for i in range(len(count)):
                    if count[i]!=0:
                        avg.append(radialProfile[i]/count[i])

                avg_fft_features = []
                num_features = 47

                for j in range(len(avg)):
                    avg_fft_features.extend(avg[j])

                if len(avg) != num_features:
                    print("			Flagged: ",image+'/'+subfolder, len(avg))
                    while len(avg_fft_features) < num_features:
                        avg_fft_features.append(0)


                # Determines the label of the image patch
                if image == "Real_Blood":
                    label = 1
                elif image == "Real_Effusion":
                    label = 0
                else:
                    image_level = int(subfolder.split("_")[0])
                    if image_level == 0:
                        label = 0
                    else:
                        label = 1

                # Adds the image's fft_features to the global array
                image_fft_features = []
                image_fft_features.append(image+'/'+subfolder)
                image_fft_features.extend(avg_fft_features)
                fft_features.append(image_fft_features)

                # Adds the image's label to the global array
                image_patch_label = []
                image_patch_label.append(image+'/'+subfolder)
                image_patch_label.append(label)
                image_labels.append(image_patch_label)

print("Finished generating fft features")

print("Images we didn't get fft features for: ", nonetype_images)

# # Prints the fft features
# X = np.zeros((len(fft_features),47),np.float64)
# for i in range(len(fft_features)):
#     X[i] = fft_features[i][1:]
#     print(fft_features[i][0], list(X[i]))


print("Saving to CSV file with PANDAS")
# Trying out saving this with pandas pd.DataFrame.to_csv method
fft_features_cols = ['ID']
fft_features_cols.extend(np.arange(47).tolist())

results_features = pd.DataFrame(fft_features, columns=fft_features_cols)
results_features.to_csv('test_split_fft_features.csv')


results_labels = pd.DataFrame(image_labels, columns=['ID', 'Label'])
results_labels.to_csv('test_split_fft_image_labels.csv')

