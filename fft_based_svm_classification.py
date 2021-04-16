
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


def extract_grouped_fft_features_and_labels_csv(fft_features_csv, image_labels_csv):
    """
    Returns 2 dictionaries containing the FFT features and labels of the corresponding patches for each ultrasound image
    in the dataset.

    1. grouped_fft_features dictionary
    - Key: the ultrasound image filename
    - Value: a list containing every image patches FFT features

    2. grouped_image_labels
    - Key: the ultrasound image filename
    - Value: the image label

    :param fft_features_csv: the file containing the 47 features for the image patches of all ultrasound images in the dataset
    :param image_labels_csv: the file containing the labels for the image patches of all ultrasound images in the dataset
    :return: two dictionaries containing all the FFT features and labels of the corresponding patches for each ultrasound image in the dataset
    """

    ID_col = fft_features_csv.iloc[:, 0]

    grouped_fft_features = {}
    grouped_image_labels = {}

    for i in range(len(ID_col)):

        img = ID_col[i]
        folder = img.split('/')[0]

        # get patch name
        patch_name = ''
        if folder == "Real_Effusion":
            splitter = '_US_'
            if ']' in img.split('/')[1]:
                splitter = ']'
            patch_name = (img.split('/')[1]).split(splitter)[0] + splitter

        elif folder == "Real_Blood":
            if 'HAM' in img.split('/')[1]:
                components = (img.split('/')[1]).split('_')
                patch_name = components[0] + '_' + components[1]
            elif 'horas' in img.split('/')[1]:
                patch_name = (img.split('/')[1]).split(']')[0] + ']'

        else:
            patch_name = img.split('_')[0]

        # retrieve fft features for the patch
        features = fft_features_csv.iloc[i]
        features = np.delete(np.array(features), 0)

        # add patch information to the 2 dictionaries
        if patch_name not in grouped_fft_features:
            grouped_fft_features[patch_name] = []
            grouped_image_labels[patch_name] = image_labels_csv.iloc[i, 1]
        grouped_fft_features[patch_name].append(features)

    return grouped_fft_features, grouped_image_labels


def preprocess_csv_data(grouped_fft_features, grouped_image_labels, type):
    """
    Returns two numpy arrays containing the combined FFT features and labels for each ultrasound image in the dataset.

    :param grouped_fft_features: a dictionary containing all the FFT features of the corresponding patches for each ultrasound image in the dataset
    :param grouped_image_labels: a dictionary containing all the labels of the corresponding patches for each ultrasound image in the dataset
    :param type: the type of combination for the FFT features
    :return: two numpy arrays containing the combined FFT features and labels for each ultrasound image in the dataset
    """

    fft_features = np.array([])
    image_labels = np.array([])
    count = 0

    for key, value in grouped_fft_features.items():
        image_fft_features = np.array(value)

        if type == 'average':
            image_fft_features = np.mean(image_fft_features, axis=0).reshape((1, 47))

        elif type == 'sum':
            image_fft_features = np.sum(image_fft_features, axis=0).reshape((1, 47))

        if count == 0:
            # print(image_fft_features)
            fft_features = np.array(image_fft_features)
            # print(image_fft_features)
        else:
            fft_features = np.concatenate((fft_features, image_fft_features), axis=0)

        label = grouped_image_labels[key]
        image_labels = np.append(image_labels, [label])
        count += 1

    image_labels = image_labels.astype('int32')

    return fft_features, image_labels


def calculate_performance_metrics(cm, dataset_label):
    """
    Calculates and prints the performance metrics given the confusion matrix.
    :param cm: the confusion matrix
    :param dataset_label: the dataset name
    """

    TN, FN, TP, FP = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]
    print(cm)

    # TP/(TP+FN) --> ratio of the correctly +ve labeled by our program to all images that really have blood in reality.
    # also known as recall
    # sensitivity = X tells us that (1-X) images out of every 10 images with blood in reality are missed by our program
    # and X images are labelled as having blood
    sensitivity = TP / (TP + FN)
    print('{} Sensitivity : {}'.format(dataset_label, sensitivity))

    # TN/(TN+FP) --> correctly -ve labeled by our program to all images that really don't have blood in reality.
    # specificity = X tells us that (1-X) images out of every 10 images with no blood in reality are miss-labelled as
    # having blood and X images are correctly labelled as having no blood
    specificity = TN / (TN + FP)
    print('{} Specificity : {}'.format(dataset_label, specificity))

    # TP/(TP+FP) --> ratio of the correctly +ve labeled by our program to all +ve labeled.
    # precision = X tells us that on average, (1-X) images out of every 10 blood labelled images are actually no blood,
    # and X images are true blood.
    precision = TP / (TP + FP)
    print('{} Precision: {}'.format(dataset_label, precision))

    # harmonic mean(average) of the precision and recall.
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    print('{} F1-score: {}'.format(dataset_label, f1_score))

    return sensitivity, specificity, precision, f1_score


def main2():
    training_fft_features = pd.read_csv('./train_split_fft_features.csv')
    training_fft_features.drop(training_fft_features.columns[[0]], axis=1, inplace=True)

    training_image_labels = pd.read_csv('./train_split_fft_image_labels.csv')
    training_image_labels.drop(training_image_labels.columns[[0]], axis=1, inplace=True)

    testing_fft_features = pd.read_csv('./test_split_fft_features.csv')
    testing_fft_features.drop(testing_fft_features.columns[[0]], axis=1, inplace=True)

    testing_image_labels = pd.read_csv('./test_split_fft_image_labels.csv')
    testing_image_labels.drop(testing_image_labels.columns[[0]], axis=1, inplace=True)

    # extract grouped fft features for training and testing
    training_grouped_fft_features, training_grouped_image_labels = extract_grouped_fft_features_and_labels_csv(
        training_fft_features, training_image_labels)
    testing_grouped_fft_features, testing_grouped_image_labels = extract_grouped_fft_features_and_labels_csv(
        testing_fft_features, testing_image_labels)

    # combine fft features by taking the sum
    training_summed_fft_features, training_summed_blood_classification_labels = preprocess_csv_data(
        training_grouped_fft_features, training_grouped_image_labels, 'sum')
    testing_summed_fft_features, testing_summed_blood_classification_labels = preprocess_csv_data(
        testing_grouped_fft_features, testing_grouped_image_labels, 'sum')

    print("total 1s in dataset:", np.count_nonzero(training_summed_blood_classification_labels == 1))
    print("total 0s in dataset:", len(training_summed_blood_classification_labels) - np.count_nonzero(training_summed_blood_classification_labels == 1))

    X_train, X_test, y_train, y_test = training_summed_fft_features, testing_summed_fft_features, training_summed_blood_classification_labels, testing_summed_blood_classification_labels
    # C=220, gamma=0.00001 --> ~70.5% test
    # C=180, gamma=0.00001 --> ~71.57% test
    svclassifier = SVC(kernel='rbf', C=200, gamma=0.00001)
    svclassifier.fit(X_train, y_train)

    # Training accuracy ################################################
    y_train_pred = svclassifier.predict(X_train)
    # print('y_train_pred', y_train_pred)
    # print('y_train_true', y_train)

    sum_train_accuracy = accuracy_score(y_train, y_train_pred)
    print('Training Accuracy of summed values:', sum_train_accuracy)

    train_cm = confusion_matrix(y_train, y_train_pred)
    calculate_performance_metrics(train_cm, 'Training')
    ###################################################################

    # Testing accuracy ################################################
    y_test_pred = svclassifier.predict(X_test)
    # print('y_test_pred', y_test_pred)
    # print('y_test_true', y_test)

    sum_test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test Accuracy of summed values:', sum_test_accuracy)

    test_cm = confusion_matrix(y_test, y_test_pred)
    calculate_performance_metrics(test_cm, 'Testing')
    ###################################################################


if __name__ == "__main__":
    main2()
    exit(0)
