import time
import random
import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import gc

'''
    Global constants
'''
PATCH_SIZE = 500
CHANNEL = 3
CLASS_NUM = 2

DROPOUT = 0.5

THRESH = 90

PIXEL_WHITE = 255
PIXEL_TH = 200
PIXEL_BLACK = 0

'''
    Load slides into memory.
'''
def read_wsi(tif_file_path, level):
    
    '''
        Identify and load slides.
    '''
    
    time_s = time.time()
    
    try:
        wsi_image = OpenSlide(tif_file_path)
        slide_w_, slide_h_ = wsi_image.level_dimensions[level]
        
        '''
            The read_region loads the target area into RAM memory, and
            returns an Pillow Image object.

            !!! Take care because WSIs are gigapixel images, which are could be 
            extremely large to RAMs.
        '''
        rgb_image_pil = wsi_image.read_region((0, 0), level, (slide_w_, slide_h_))
        print("width, height:", rgb_image_pil.size)

        '''
            !!! It should be noted that:
            1. np.asarray() / np.array() would switch the position 
            of WIDTH and HEIGHT in shape.

            Here, the shape of $rgb_image_pil is: (WIDTH, HEIGHT, CHANNEL).
            After the np.asarray() transformation, the shape of $rgb_image is: 
            (HEIGHT, WIDTH, CHANNEL).

            2. The image here is RGBA image, in which A stands for Alpha channel.
            The A channel is unnecessary for now and could be dropped.
        '''
        rgb_image = np.asarray(rgb_image_pil)
        print("transformed:", rgb_image.shape)
        
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None

    time_e = time.time()
    
    print("Time spent on loading", tif_file_path, ": ", (time_e - time_s))
    
    return wsi_image, rgb_image, (slide_w_, slide_h_)

'''
    Convert RGBA to RGB, HSV and GRAY.
'''
def construct_colored_wsi(rgba_):

    '''
        This function splits and merges R, G, B channels.
        HSV and GRAY images are also created for future segmentation procedure.
    '''
    r_, g_, b_, a_ = cv2.split(rgba_)
    
    wsi_rgb_ = cv2.merge((r_, g_, b_))
    
    wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_BGR2GRAY)
    wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_BGR2HSV)
    
    return wsi_rgb_, wsi_gray_, wsi_hsv_

'''
'''
def get_contours(cont_img, rgb_image_shape):
    
    '''
    Args:
        - cont_img: images with contours, these images are in np.array format.
        - rgb_image_shape: shape of rgb image, (HEIGHT, WIDTH).

    Returns: 
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of valid region coordinates (contours squeezed);
        - contours: List of valid regions (coordinates);
        - mask: binary mask array;

        ! It should be noticed that the shape of mask array is: (HEIGHT, WIDTH, CHANNEL).
    '''
    
    print('contour image: ',cont_img.shape)
    
    contour_coords = []
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    for contour in contours:
        contour_coords.append(np.squeeze(contour))
        
    mask = np.zeros(rgb_image_shape, np.uint8)
    
    print('mask shape', mask.shape)
    cv2.drawContours(mask, contours, -1, \
                    (PIXEL_WHITE, PIXEL_WHITE, PIXEL_WHITE),thickness=-1)
    
    return boundingBoxes, contour_coords, contours, mask

'''
    Perform segmentation and get contours.
'''
def segmentation_hsv(wsi_hsv_, wsi_rgb_):
    '''
    Args:
        - wsi_hsv_: HSV images.
        - wsi_rgb_: RGB images.

    Returns: 
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of arrays. Each array stands for a valid region and 
        contains contour coordinates of that region.
        - contours: Almost same to $contour_coords;
        - mask: binary mask array;

        ! It should be noticed that:
        1. The shape of mask array is: (HEIGHT, WIDTH, CHANNEL);
        2. $contours is unprocessed format of contour list returned by OpenCV cv2.findContours method.
        
        The shape of arrays in $contours is: (NUMBER_OF_COORDS, 1, 2), 2 stands for x, y;
        The shape of arrays in $contour_coords is: (NUMBER_OF_COORDS, 2), 2 stands for x, y;

        The only difference between $contours and $contour_coords is in shape.
    '''
    print("HSV segmentation: ")
    contour_coord = []
    
    '''
        Here we could tune for better results.
        Currently 20 and 200 are lower and upper threshold for H, S, V values, respectively. 
    '''
    lower_ = np.array([20,20,20])
    upper_ = np.array([200,200,200]) 

    # HSV image threshold
    thresh = cv2.inRange(wsi_hsv_, lower_, upper_)
    
    try:
        print("thresh shape:", thresh.shape)
    except:
        print("thresh shape:", thresh.size)
    else:
        pass
    
    '''
        Closing
    '''
    print("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8) 
    image_close = cv2.morphologyEx(np.array(thresh),cv2.MORPH_CLOSE, close_kernel)
    print("image_close size", image_close.shape)

    '''
        Openning
    ''' 
    print("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    print("image_open size", image_open.size)

    print("Getting Contour: ")
    bounding_boxes, contour_coords, contours, mask \
    = get_contours(np.array(image_open), wsi_rgb_.shape)
      
    return bounding_boxes, contour_coords, contours, mask



