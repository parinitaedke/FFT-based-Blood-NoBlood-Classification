# Original creator: Mauro
# Modified by: Parinita Edke
# Date of first modification: 18th February, 2021

import cv2
import segmentation_models_pytorch as smp
import torch
import albumentations as albu
import numpy as np
import pathlib
import glob

# Use CPU
DEVICE = torch.device("cpu")

# Size of patch
PATCH_SIZE = 64

# Folder to store the patches
OUTPUT_FOLDER = './patches_sample/'
BLOOD_FOLDER = './train_real_blood_patches/'
EFFUSION_FOLDER = './train_real_effusion_patches/'

pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(BLOOD_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(EFFUSION_FOLDER).mkdir(parents=True, exist_ok=True)


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
    
    print("	Before input_tensor")
    print("	Input type:", type(input))
    input_tensor = torch.tensor(SEG_TRANSFORMS(image=input)['image'])
    print("	After input_tensor")

    # Get Segmentation
    with torch.no_grad():
        mask = model(input_tensor.unsqueeze(0))
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        mask = mask.squeeze().numpy() * 255

    # Post Processing - Upscaling
    resized = cv2.resize(np.uint8(mask), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_AREA)

    return resized


def patches(image, segmentation, folder_name, level):
    """
       Function to capture patches of the original image based on the segmentation.

       image: Original image (numpy array)
       segmentation: Binarized segmentation (numpy array)
    """

    # Get contours of segmenation blob
    contours, _ = cv2.findContours(segmentation, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    idx = 0 # Number of patch

    for cnt in contours: # For every blob of segmentation

        # Calculate a rectangle sorrounding the blob
        x,y,w,h = cv2.boundingRect(cnt)

        # Go through rectangle in PATCH_SIZExPATCH_SIZE incrementing steps
        for y_i in range(y, y+h, PATCH_SIZE): # Vertical
            for x_i in range(x, x+w, PATCH_SIZE): # Horizontal
                # Patch to validate if area is on the segmentation
                patch_seg = segmentation[y_i:y_i+PATCH_SIZE, x_i:x_i+PATCH_SIZE]

                if patch_seg.sum() > 0: # If there are pixels of segmentation in this patch
                    # Patch of the original image
                    patch = image[y_i:y_i+PATCH_SIZE, x_i:x_i+PATCH_SIZE]
                    # Save patch
                    #cv2.imwrite(OUTPUT_FOLDER + str(idx) + '.png', patch)
                    cv2.imwrite(OUTPUT_FOLDER + folder_name + '/' + str(level) + '_' + str(idx) + '.png', patch)
                    idx += 1
    print("	Num patches for level {}: {}".format(level, idx))

###################################################################################################################
# Rajshree's patches function
def get_64x64_patches_arnd_effusion(image, segmentation, folder_name, level, output_folder):
    """
       Function to get 64x64 mutually exclusive patches from effuction area only

       image: Original image (numpy array) [single channel input us image]
       segmentation: Binarized segmentation (numpy array) [single channel effusion mask]
    """

    x, y, w, h = cv2.boundingRect(segmentation)
    # print(x, y, w, h)
    w_new = w
    if w < PATCH_SIZE:
        w_new = PATCH_SIZE
    
    h_new = h
    if h < PATCH_SIZE:
        h_new = PATCH_SIZE

    # Number of patches
    idx = 0

    for i in range(int(w_new/PATCH_SIZE)):
        for j in range(int(h_new/PATCH_SIZE)):
            if np.sum(segmentation[y + j*PATCH_SIZE:y + j*PATCH_SIZE + PATCH_SIZE, x + i*PATCH_SIZE:x + i*PATCH_SIZE+PATCH_SIZE])>0:
                patch = image[y + j*PATCH_SIZE:y + j*PATCH_SIZE + PATCH_SIZE, x + i*PATCH_SIZE:x + i*PATCH_SIZE+PATCH_SIZE]

                # Save patch
                print("Saved image path: ", output_folder + folder_name + '/' + str(level) + '_' + str(idx) + '.png')
                cv2.imwrite(output_folder + folder_name + '/' + str(level) + '_' + str(idx) + '.png', patch)
                idx += 1

    print("     Num patches for level {}: {}".format(level, idx))

###################################################################################################################


def main(image_path, folder_name, level, output_folder):
    """
       Main function to get patches from a single image
    """
    # image_path = '../Real_Effusion/m4605_a4703_s4743_1_13_US_.png'
    
    print('Arguments in main:', image_path, folder_name, level)
    print('	Image path type in main:', type(image_path))

    image_path = image_path.rstrip()

    # Read image
    print('	Reading Image...')
    image = cv2.imread(image_path)
    print("	Image type in main:", type(image))

    # Segment the image
    print('	Segmenting...')
    segmentation = segment(image)

    # Make patches
    print('	Patching using Rajshrees function...')
    # patches(image, segmentation, folder_name, level)
    get_64x64_patches_arnd_effusion(image, segmentation, folder_name, level, output_folder)

    print('	Done!')


def synthetic_data_patches():
    """
    Generates the patches for the synthetic data.
    """
    dataset = pathlib.Path('../Synthetic_Knees/dataset_level.csv')

    with dataset.open() as file:
        missing_imgs = []
        for cnt, line in enumerate(file):
            if cnt == 0:
                continue

            components = line.split(',')
            path_splits = components[1].split('/')
            
            folder_path = components[1]
            folder_name = ''
            for i in range(len(path_splits)):
                if i == 7:
                    folder_name = path_splits[i]
            # so now we have: folder name, different level images, and the path to the images
            # we just iterate through the images and then save the patches to the folder
            
            # we created a subfolder for this folder_name within OUTPUT_FOLDER
            img_folder_path = pathlib.Path(OUTPUT_FOLDER) / folder_name
            pathlib.Path(img_folder_path).mkdir(parents=True, exist_ok=True)
            # print(img_folder_path)

            # 2 -> LVL 0, 3 -> LVL 1, 4 -> LVL2, 5 -> LVL 3, 6 -> LVL 4
            for j in range(len(components[2:])):
                if j not in [0, 2]:
                    continue

                img = components[2:][j]

                # if Level 0/2 image is not there, we continue
                if len(img.rstrip()) == 0:
                    # print('################# Dont have image for', folder_name, 'level', j, '#################')
                    missing_imgs.append(folder_name + '_' + str(j))
                    continue
                else:
                    # now we have the full image path, the level of the image, and created the folder where this image will be stored
                    main(components[1]+img, folder_name, j, OUTPUT_FOLDER)

    # print("Missing images:", missing_imgs)


def real_data_patches(real_dataset, folder_name, output_folder):
    """
    Generates the patches for the real patient data.
    """
    # we created a subfolder for this folder_name within OUTPUT_FOLDER
    img_folder_path = pathlib.Path(output_folder) / folder_name
    pathlib.Path(img_folder_path).mkdir(parents=True, exist_ok=True)
    i = 0
    for path in real_dataset:
        image_path = str(path)
        # print(image_path)
        # retrieve file name (file name replaces level as there is only 1 level per image)
        if (str(path).split('/')[-1])[-3:] == 'png':
            f_name = (str(path).split('/')[-1]).split('.png')[0].rstrip()
        else:
            f_name = (str(path).split('/')[-1]).split('.jpg')[0].rstrip()

        # print("Image: ", f_name)

        main(image_path, folder_name, f_name, output_folder)
        # print(i)
        i += 1


if __name__ == "__main__":
    # main('temp', '4A', 3)

    print("Patching images...")

    # Depending on what images you want to segment, call the appropriate function.
    # Synthetic images were stored in a different way than the real patient data and so, to patch the synthetic images,
    # call the synthetic_data_patches() function

    synthetic_data_patches()

    # Similarly to patch the real patient data, call the real_data_patches() function
    real_blood_dataset = pathlib.Path('../../RB_split/Split_dataset/train/Real_Blood').glob('*')
    real_effusion_dataset = pathlib.Path('../../RE_split/Split_dataset/train/Real_Effusion').glob('*')

    real_data_patches(real_blood_dataset, "Real_Blood", BLOOD_FOLDER)
    real_data_patches(real_effusion_dataset, "Real_Effusion", EFFUSION_FOLDER)
    print("Done!")
