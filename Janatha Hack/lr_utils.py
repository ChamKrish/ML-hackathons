import numpy as np
import pandas as pd
from matplotlib import image

def load_csv_dataset():
    
    train_dataset = pd.read_csv(r'train.csv')
    images = convert(train_dataset["image_names"][:])
    train_set_x_orig = np.array(images)
    train_set_y_orig = np.array(train_dataset["emergency_or_not"][:])
    
    test_dataset = pd.read_csv(r'sample_submission_yxjOnvz.csv')
    images = convert(test_dataset["image_names"][:])
    test_set_x_orig = np.array(images)
    test_set_y_orig = np.array(test_dataset["emergency_or_not"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def convert(image_names):
    x_dataset = []
    for i in image_names:
        im = image.imread('images/' + i)
        x_dataset.append(im)
    return x_dataset