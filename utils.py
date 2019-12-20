import numpy as np
import cv2


def preprocess_img(fname, im_shape):
    """
    ### Arguments
        fname: image path
        imshape: the shape you want your image to be resized into, expects tuple with width and height ex. (100, 100).
    ##### Returns: Resized and normalized image in Grayscale with shape (width, height, 1)
    #### Usage: Use it to resize your images. We are using it by iteratively passing image paths into it to get back preprocessed images. 
    """
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=im_shape)
    img = np.array(img)
    img = img.reshape(im_shape[0], im_shape[1], 1)
    img = img / 255.0

    return img

def convert_to_one_hot(Y, C):
    """
    ### Arguments
        Y: Array with you labels
        C: The number of classes
    ##### Returns: returns one_hot_encoded labels
    """
    Y = np.eye(C)[Y.reshape(-1)]
    return Y