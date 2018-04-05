'''
    Written in Python 3.5
'''

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
    Global variables / constants
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
    This newly added parameter defines how many parts we will split the WSI into.
    For example, SPLIT=4 means we will process 16=4*4 parts of WSI in turn.
'''
SPLIT = 4

level = 1
mag_factor = pow(2, level)

'''
    !!!! It should be noticed with great caution that:

    The coordinates in the bounding boxes/contours are scaled ones.
    For example, when we choose level3 (5x magnification), the magnification factor 
    would be 8 (2^3).

    If we selected (200, 300) in level3 scale, the corresponding level0 coordinates 
    should be (200 * 8, 300 * 8). 

    The coordinates used in functions below are in selected level scale, which means:
    (COORDS_X_IN_LEVEL0 / mag_factor, COORDS_Y_IN_LEVEL0 / mag_factor).

    But the read_region() method of OpenSlide object performs in level0 scale, so 
    transformation is needed when invoking read_region().
'''

def openSlide_init(tif_file_path, level):
    '''
        Identifies the slide and initializes OpenSlide object.
    '''
    try:
        wsi_obj = OpenSlide(tif_file_path)

    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None
    else:
        slide_w_, slide_h_ = wsi_obj.level_dimensions[level]
        print('level' + str(level), 'size(w, h):', slide_w_, slide_h_)
        
        return wsi_obj

'''
    Load selected parts of the slides into memory.
'''
def read_wsi(wsi_obj, level, mag_factor, sect):
    
    '''
        Identify and load slides.
        Args:
            wsi_obj: OpenSlide object;
            level: magnification level;
            mag_factor: pow(2, level);
            sect: string, indicates which part of the WSI. For example:
            sect='12':
             _ _ _ _
            |_|_|_|_|                  
            |_|_|_|_|
            |_|*|_|_|
            |_|_|_|_|   
            
            '01':
             _ _ _ _
            |_|_|_|_|                  
            |*|_|_|_|
            |_|_|_|_|
            |_|_|_|_| 

        Returns:
            - rgba_image: WSI image loaded, NumPy array type.
    '''
    
    time_s = time.time()
            
    '''
        The read_region loads the target area into RAM memory, and
        returns an Pillow Image object.

        !! Take care because WSIs are gigapixel images, which are could be 
        extremely large to RAMs.

        Load the whole image in level < 3 could cause failures.
    '''

    # Here we load the whole image from (0, 0), so transformation of coordinates 
    # is not skipped.

    # level1 dimension
    width_whole, height_whole = wsi_obj.level_dimensions[level]
    print(width_whole, height_whole)

    # section size after split
    width_split, height_split = width_whole // SPLIT, height_whole // SPLIT
    print(width_split, height_split)

    delta_x = int(sect[0]) * width_split
    delta_y = int(sect[1]) * height_split

    '''
        Be aware that the first arg of read_region is a tuple of coordinates in 
        level0 reference frame.
    '''
    rgba_image_pil = wsi_obj.read_region((delta_x * width_split * mag_factor, \
                                          delta_y * height_split * mag_factor), \
                                          level, (width_split, width_split))
    print("width, height:", rgba_image_pil.size)

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
    rgba_image = np.asarray(rgba_image_pil)
    print("transformed:", rgba_image.shape)

    time_e = time.time()
    
    print("Time spent on loading: ", (time_e - time_s))
    
    return rgba_image

'''
    Convert RGBA to RGB, HSV and GRAY.
'''
def construct_colored_wsi(rgba_):

    '''
        This function splits and merges R, G, B channels.
        HSV and GRAY images are also created for future segmentation procedure.

        Args:
            - rgba_: Image to be processed, NumPy array type.

    '''
    r_, g_, b_, a_ = cv2.split(rgba_)
    
    wsi_rgb_ = cv2.merge((r_, g_, b_))
    
    wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_RGB2GRAY)
    wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)
    
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

        !!! It should be noticed that the shape of mask array is: (HEIGHT, WIDTH, CHANNEL).
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
    This func is designed to remove background of WSIs. 

    Args:
        - wsi_hsv_: HSV images.
        - wsi_rgb_: RGB images.

    Returns: 
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of arrays. Each array stands for a valid region and 
        contains contour coordinates of that region.
        - contours: Almost same to $contour_coords;
        - mask: binary mask array;

        !!! It should be noticed that:
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
    
        !!! It should be noted that the threshold values here highly depends on the dataset itself.
        Thresh value could vary a lot among different datasets.
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


'''
    Extract Valid patches.
'''
def construct_bags(wsi_, wsi_rgb, contours, mask, level, mag_factor, PATCH_SIZE):
    
    '''
    Args:
        To-do.

    Returns: 
        - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, CHANNEL]
        - patches_coords: coordinates of patches: (x_min, y_min). The bouding box of the patch
        is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
    '''

    patches = []
    patches_coords = []

    start = time.time()
    
    '''
        !!! 
        Currently we select only the first 5 regions, because there are too many small areas and 
        too many irrelevant would be selected if we extract patches from all regions.

        And how many regions from which we decide to extract patches is 
        highly related to the SEGMENTATION results.

    '''
    contours_ = sorted(contours, key = cv2.contourArea, reverse = True)
    contours_ = contours_[:5]

    for i, box_ in enumerate(contours_):

        box_ = cv2.boundingRect(np.squeeze(box_))
        print('region', i)
        
        '''

        !!! Take care of difference in shapes:

            Coordinates in bounding boxes: (WIDTH, HEIGHT)
            WSI image: (HEIGHT, WIDTH, CHANNEL)
            Mask: (HEIGHT, WIDTH, CHANNEL)

        '''

        b_x_start = int(box_[0])
        b_y_start = int(box_[1])
        b_x_end = int(box_[0]) + int(box_[2])
        b_y_end = int(box_[1]) + int(box_[3])
        
        '''
            !!!
            step size could be tuned for better results.
        '''

        X = np.arange(b_x_start, b_x_end, step=PATCH_SIZE // 2)
        Y = np.arange(b_y_start, b_y_end, step=PATCH_SIZE // 2)        
        
        print('ROI length:', len(X), len(Y))
        
        for h_pos, y_height_ in enumerate(Y):
        
            for w_pos, x_width_ in enumerate(X):

                # Read again from WSI object wastes tooooo much time.
                # patch_img = wsi_.read_region((x_width_, y_height_), level, (PATCH_SIZE, PATCH_SIZE))
                
                '''
                    !!! Take care of difference in shapes
                    Here, the shape of wsi_rgb is (HEIGHT, WIDTH, CHANNEL)
                    the shape of mask is (HEIGHT, WIDTH, CHANNEL)
                '''
                patch_arr = wsi_rgb[y_height_: y_height_ + PATCH_SIZE,\
                                    x_width_:x_width_ + PATCH_SIZE,:]            
                print("read_region (scaled coordinates): ", x_width_, y_height_)

                width_mask = x_width_
                height_mask = y_height_                
                
                patch_mask_arr = mask[height_mask: height_mask + PATCH_SIZE, \
                                      width_mask: width_mask + PATCH_SIZE]

                print("Numpy mask shape: ", patch_mask_arr.shape)
                print("Numpy patch shape: ", patch_arr.shape)

                try:
                    bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)
                
                except Exception as err:
                    print('Out of the boundary')
                    pass
                    
#                     f_ = ((patch_arr > PIXEL_TH) * 1)
#                     f_ = (f_ * PIXEL_WHITE).astype('uint8')

#                     if np.mean(f_) <= (PIXEL_TH + 40):
#                         patches.append(patch_arr)
#                         patches_coords.append((x_width_, y_height_))
#                         print(x_width_, y_height_)
#                         print('Saved\n')

                else:
                    bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
                    white_pixel_cnt = cv2.countNonZero(bitwise_grey)

                    '''
                        Patches whose valid area >= 25% of total area is considered
                        valid and selected.
                    '''

                    if white_pixel_cnt >= ((PATCH_SIZE ** 2) * 0.25):

                        patches.append(patch_arr)
                        patches_coords.append((x_width_, y_height_))
                        print(x_width_, y_height_)
                        print('Saved\n')

                    else:
                        print('Did not save\n')

    end = time.time()
    print("Time spent on patch extraction: ",  (end - start))

    # patches_ = [patch_[:,:,:3] for patch_ in patches] 
    print("Total number of patches extracted:", len(patches))
    
    return patches, patches_coords

'''
    Save patches to disk.
'''
def save_to_disk(patches, patches_coords, mask, slide_, level):
    
    '''
        The paths should be changed
    '''
    
    case_name = slide_.split('/')[-1].split('.')[0]
    prefix_dir = './dataset_patches/' + case_name + '/level' + str(level) + '/' 
    patch_array_dst = './dataset_patches/' + case_name + '/level' + str(level) + '/patches/'
    
    patch_coords_dst = './dataset_patches/' + case_name + '/level' + str(level) + '/'
    array_file = patch_array_dst + 'patch_'
    
    coords_file = patch_coords_dst + 'patch_coords.csv'
    mask_file = patch_coords_dst + 'mask'

    if not os.path.exists(patch_array_dst):
        os.makedirs(patch_array_dst)
        print('mkdir', patch_array_dst)

    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
        print('mkdir', prefix_dir)
    
    print('Path: ', array_file)
    print('Path: ', coords_file)
    print('Path: ', mask_file)
    print('Number of patches: ', len(patches_coords))
    print(patches_coords[:5])
    
    '''
        Save coordinates to the disk. Here we use pandas DataFrame to organize 
        and save coordinates.
    '''

    df1_ = pd.DataFrame([x[0] for x in patches_coords], columns = ["coord_x"])
    df1_["coord_y"] = [y[1] for y in patches_coords]
    df1_.to_csv(coords_file, encoding='utf-8', index=False)
    
    '''
    Save patch arrays to the disk
    '''
    # patch_whole = np.array(patches1).shape

    for i, patch_ in enumerate(patches):
        x_, y_ = patches_coords[i]
        patch_name = array_file + str(i) + '_' + str(x_) + '_' + str(y_)
        
        np.save(patch_name, np.array(patch_))
        im = Image.fromarray(patch_)
        im.save(patch_name + '.jpeg')
        
    # Save whole patches: convert list of patches to array.
    # shape: (NUMBER_OF_PATCHES, PATCH_WIDTH, PATCH_HEIGHT, CHANNEL)

    patch_whole = prefix_dir + 'patch_whole'
    np.save(patch_whole, np.array(patches))
    
    '''
    Save mask file to the disk
    '''
    np.save(mask_file, mask)


'''
    The whole pipeline of extracting patches.
'''
def extract_(slide_, level, mag_factor):
    '''
    Args:
        slide_: path to target slide.
        level: magnification level. 
        mag_factor: pow(2, level).

    Returns: 
        To-do.
    '''

    start = time.time()
    
    wsi_, rgba_, shape_ = read_wsi(slide_, level)
    wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba3_)

    print('Transformed shape: (height, width, channel)')
    print("WSI HSV shape: ", wsi_hsv_.shape)
    print("WSI RGB shape: ", wsi_rgb_.shape)
    print("WSI GRAY shape: ", wsi_gray_.shape)
    print('\n')

    del rgba_
    gc.collect()

    bounding_boxes, contour_coords, contours, mask \
    = segmentation_hsv(wsi_hsv_, wsi_rgb_)

    del wsi_hsv_
    gc.collect()

    patches, patches_coords = construct_bags(wsi_, wsi_rgb_, contours, mask, \
                                            level, mag_factor, PATCH_SIZE)

    # save_to_disk(patches, patches_coords, mask, slide_, level)
    
    end = time.time()
    print("Time spent on patch extraction: ",  (end - start))
    
    return patches, patches_coords, mask
