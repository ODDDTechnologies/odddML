import numpy as np
import os
import cv2



class alignDocs(object):
    """
    # Arguments
        no_features: number of features to align
        GOOD_MATCH_PERCENT: the match percentage to create an aligned image, defaults to 0.15 not recommended to change it if you don't know what you are doing... 
    #### Usage: use the methods .work_image to align one image based on a reference image and .work_images for a directory of images. 
    """ 
    
    def __init__(self, no_features, GOOD_MATCH_PERCENT=0.15):
        self.no_features = no_features
        self.GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT

    def alignImages(self, im1, im2):
            # Convert images to grayscale 
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.no_features)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, _ = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h
    
    def work_images(self, ref_img, src_dir, out_dir):
        """
        # Arguments
            refimg: the reference image that you'll use as a template
            srcdir: the directory of the input images pass it as a subdir, ex. ("directory_1/")
            outdir: the directory you want to output your images into, same syntax as src_dir 
        #### Usage: use this method to align a whole directory of images
        """ 
        refFilename = ref_img
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
        all_files = [file for file in os.listdir(src_dir) if file.endswith(".jpeg")]
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # Read image to be aligned
        for each in all_files:
            imFilename = each
            print("Reading image to align : ", imFilename)
            
            im = cv2.imread(os.path.join(src_dir, imFilename), cv2.IMREAD_COLOR)
            #print("Aligning images ...")
            # Registered image will be resotred in imReg. 
            # The estimated homography will be stored in h. 
            imReg, h = self.alignImages(im, imReference)

            # Write aligned image to disk. 
            outFilename = each
            print("Saving aligned image : ", outFilename) 
            cv2.imwrite(os.path.join(out_dir, outFilename), imReg)


    def work_image(self, ref_img, image):
        """
        # Arguments
            refimg: the reference image that you'll use as a template
            image: the image that you want to align with the refimg
        #### Usage: use this method to align one image each time.
        """  
        refFilename = ref_img
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        im = cv2.imread(image, cv2.IMREAD_COLOR)

        imReg, h = self.alignImages(im, imReference)
        cv2.imwrite(image, imReg)
        print("saved")
        


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

if __name__ == "__main__":
    pass