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
    Load slides into memory
'''
def read_wsi(tif_file_path, level):
    
    """
        Identify and load slides.
    """
    
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

