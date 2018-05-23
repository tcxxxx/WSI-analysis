'''
    Written in Python 3.5
'''
import time
import random
import os
import cv2
import numpy as np
import pandas as pd
import gc

from PIL import Image
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import xml.etree.cElementTree as ET
from shapely.geometry import box, Point, Polygon

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from ast import literal_eval

import pickle

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

        Returns:
            - wsi_obj: OpenSlide object to the target WSI.
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
    print("level1 dimension (width, height): ", width_whole, height_whole)

    # section size after split
    width_split, height_split = width_whole // SPLIT, height_whole // SPLIT
    print("section size (width, height): ", width_split, height_split)

    '''
        Be aware that the first arg of read_region is a tuple of coordinates in 
        level0 reference frame.
    '''
    delta_x = int(sect[0]) * width_split
    delta_y = int(sect[1]) * height_split

    rgba_image_pil = wsi_obj.read_region((delta_x * mag_factor, \
                                          delta_y * mag_factor), \
                                          level, (width_split, height_split))

    print("rgba image dimension (width, height):", rgba_image_pil.size)

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
    
    print("Time spent on loading WSI section into memory: ", (time_e - time_s))
    
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

        Returns:
            - wsi_rgb_: RGB image, NumPy array type.
            - wsi_gray_: Grayscale image, NumPy array type.
            - wsi_hsv_: HSV image, NumPy array type.

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
    
    print('contour image dimension: ',cont_img.shape)
    
    contour_coords = []
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    for contour in contours:
        contour_coords.append(np.squeeze(contour))
        
    mask = np.zeros(rgb_image_shape, np.uint8)
    
    print('mask image dimension: ', mask.shape)
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
    print("HSV segmentation step")
    contour_coord = []
    
    '''
        Here we could tune for better results.
        Currently 20 and 200 are lower and upper threshold for H, S, V values, respectively. 
    
        !!! It should be noted that the threshold values here are highly dependent on 
        the dataset itself. Thresh values could vary a lot among different datasets.

        How to find the thresholding values that fit the dataset most is an important topic.
    '''
    lower_ = np.array([20,20,20])
    upper_ = np.array([200,200,200]) 

    # HSV image threshold
    thresh = cv2.inRange(wsi_hsv_, lower_, upper_)
    
    '''
        Closing Step
    '''
    # print("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8) 
    image_close = cv2.morphologyEx(np.array(thresh),cv2.MORPH_CLOSE, close_kernel)
    # print("image_close size", image_close.shape)

    '''
        Openning Step
    ''' 
    # print("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    # print("image_open size", image_open.size)

    # print("Getting Contour: ")
    bounding_boxes, contour_coords, contours, mask \
    = get_contours(np.array(image_open), wsi_rgb_.shape)
      
    return bounding_boxes, contour_coords, contours, mask


'''
    Extract patches which are considered valid in the segmentation step. 
'''
def construct_bags(wsi_obj, wsi_rgb, contours, mask, level, \
                   mag_factor, sect, patch_size, split_num):
    
    '''
    Args:
        - wsi_obj: 
        - wsi_rgb:
        - contours:
        - mask:
        - level:
        - mag_factor:
        - sect
        - patch_size:
        - split_num: 

    Returns: 
        - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, CHANNEL]

        - patches_coords: coordinates of patches, (x_min, y_min). 
        The bouding box of the patch is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
    
    '''

    patches = list()
    patches_coords = list()
    patches_coords_local = list()

    start = time.time()
    
    # level1 dimension
    width_whole, height_whole = wsi_obj.level_dimensions[level]
    width_split, height_split = width_whole // split_num, height_whole // split_num
    # print(width_whole, height_whole)

    # section size after split
    # print(int(sect[0]), int(sect[1]))
    delta_x = int(sect[0]) * width_split
    delta_y = int(sect[1]) * height_split
    # print("delta:", delta_x, delta_y)

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
        # print('region', i)
        # 
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
            step size could be tuned for better results, and step size will greatly affect 
            the number of patches extracted.
        '''

        # step size: PATCH_SIZE / 2 -> PATCH_SIZE

        X = np.arange(b_x_start, b_x_end, step=patch_size)
        Y = np.arange(b_y_start, b_y_end, step=patch_size)        
        
        # print('ROI length:', len(X), len(Y))
        
        for h_pos, y_height_ in enumerate(Y):
        
            for w_pos, x_width_ in enumerate(X):

                # Read again from WSI object wastes tooooo much time.
                # patch_img = wsi_.read_region((x_width_, y_height_), level, (patch_size, patch_size))
                
                '''
                    !!! Take care of difference in shapes
                    Here, the shape of wsi_rgb is (HEIGHT, WIDTH, CHANNEL)
                    the shape of mask is (HEIGHT, WIDTH, CHANNEL)
                '''
                patch_arr = wsi_rgb[y_height_: y_height_ + patch_size,\
                                    x_width_:x_width_ + patch_size,:]            
                # print("read_region (scaled coordinates): ", x_width_, y_height_)

                width_mask = x_width_
                height_mask = y_height_                
                
                patch_mask_arr = mask[height_mask: height_mask + patch_size, \
                                      width_mask: width_mask + patch_size]

                # print("Numpy mask shape: ", patch_mask_arr.shape)
                # print("Numpy patch shape: ", patch_arr.shape)

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

                    if white_pixel_cnt >= ((patch_size ** 2) * 0.5):

                        if (patch_arr.shape[0], patch_arr.shape[1])  == \
                        (patch_size, patch_size):

                            patches.append(patch_arr)
                            patches_coords.append((x_width_ + delta_x , 
                                                   y_height_ + delta_y))
                            patches_coords_local.append((x_width_, y_height_))

                            # print("global:", x_width_ + delta_x, y_height_ + delta_y)
                            # print("local: ", x_width_, y_height_)
                            # print('Saved\n')

                    else:
                        pass
                        # print('Did not save\n')

    # end = time.time()
    # print("Time spent on patch extraction: ",  (end - start))

    # patches_ = [patch_[:,:,:3] for patch_ in patches] 
    print("Total number of patches extracted: ", len(patches))
    
    return patches, patches_coords, patches_coords_local

'''
    Parse xml annotation files.
'''
def parse_annotation(anno_path, level, mag_factor):
    
    '''
    Args:
        - anno_path:
        - level:
        - mag_factor:

    Returns:
        
    '''

    polygon_list = list()
    anno_list = list()
    anno_local_list = list()

    tree = ET.ElementTree(file = anno_path)

    for an_i, crds in enumerate(tree.iter(tag='Coordinates')):
        '''
            In this loop, we process one seperate area of annotation at a time.
        '''
        # print("annotation area #%d", an_i)

        # node_list = list()
        node_list_=list()

        for coor in crds:
            '''
                Here (x, y) uses global reference in the chosen level, which means
                (x, y) indicates the location in the whole patch, rather than in 
                splited sections.
            '''
            x = int(float(coor.attrib['X'])) / mag_factor
            y = int(float(coor.attrib['Y'])) / mag_factor

            x = int(x)
            y = int(y)

            # node_list.append(Point(x,y))
            node_list_.append((x,y))
        
        anno_list.append(node_list_)

        if len(node_list_) > 2:
            polygon_ = Polygon(node_list_)
            polygon_list.append(polygon_)
    
    return polygon_list, anno_list
    

'''
    Save patches to disk.
'''
def save_to_disk(patches, patches_coords, tumor_dict, mask, 
    slide_, level, current_section):
    
    '''
        The paths should be changed to your own paths.
    '''
    case_name = slide_.split('/')[-1].split('.')[0]

    prefix_dir = './dataset_patches/' + case_name + \
                 '/level' + str(level) + '/' + current_section + '/'

    patch_array_dst = './dataset_patches/' + case_name + \
                      '/level' + str(level) + '/' + current_section + '/patches/' 

    patch_coords_dst = './dataset_patches/' + case_name + \
                       '/level' + str(level) + '/' + current_section + '/'

    array_file = patch_array_dst + 'patch_'
    
    coords_file = patch_coords_dst + 'patch_coords' + current_section + '.csv'
    mask_file = patch_coords_dst + 'mask'

    if not os.path.exists(patch_array_dst):
        os.makedirs(patch_array_dst)
        print('mkdir ', patch_array_dst)

    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
        print('mkdir ', prefix_dir)
    
    print('Path: ', array_file)
    print('Path: ', coords_file)
    print('Path: ', mask_file)
    print('Number of patches: ', len(patches_coords))
    
    '''
        Save coordinates to the disk. Here we use pandas DataFrame to organize 
        and save coordinates.
    '''

    df1_ = pd.DataFrame([coord[0] for coord in patches_coords], columns = ["coord_x"])
    df1_["coord_y"] = [coord[1] for coord in patches_coords]
    
    if tumor_dict == None:

        df1_["tumor_area"] = [0 for coord in patches_coords]
        df1_["tumor_%"] = [0 for coord in patches_coords]
    
    else:
        df1_["tumor_area"] = [tumor_dict[coord] for coord in patches_coords]
        df1_["tumor_%"] = [tumor_dict[coord] / (PATCH_SIZE * PATCH_SIZE) \
                       for coord in patches_coords]

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

    # patch_whole = prefix_dir + 'patch_whole' + current_section
    # np.save(patch_whole, np.array(patches))
    
    '''
        Save mask file to the disk. Uncomment if mask file is needed.
    '''
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_file + '.jpeg')


'''
    Utils
'''
def calculate_polygon(poly_, start_x, start_y, patch_width, 
                      patch_height, MultiPolygon_flag, draw=False):
    
    """Calculates the area of intersection one polygon area.
    
    The commented section below could plot the annotation and patch box at the same time. 
    However, this process may slow down the whole pipeline.
    
    Args:
        poly_: a shapely Polygon object.
        
        start_x: min(patch.coordinates.x). (start_x, start_y) is the left-bottom point.
        start_x: min(patch.coordinates.y). 
        
        patch_width: width of the patch.
        patch_height: height of the patch.
        
        MultiPolygon_flag: 
            if True, this polygon area is only part the annotation, which means the whole 
            annotation includes multiple polygon areas.
            if False, this polygon area is one whole annotation area.
    
    Returns:
        the size and coordinates of the intersection area (patch ∩ tumor area).
        Returns inter_area = 0, isect_x = -1, isect_y = -1 when (patch ∩ tumor area) == None. 
    
    Raises:
        None.
    """
    
    isect_ = []
    
    # (x, y): coordinates of the tumor area
    x_, y_ = poly_.exterior.coords.xy
    
    # bounding box of the tumor area
    bound_ = box(poly_.bounds[0], poly_.bounds[1], poly_.bounds[2], poly_.bounds[3])
    
    # patch box
    patch_ = box(start_x, start_y, start_x + patch_width, start_y + patch_height)
    
    # here we calculate the intersection area
    intersection_ = poly_.intersection(patch_)
    
    if not intersection_.is_empty:
        # isect_x, isect_y = intersection_.exterior.coords.xy
        
        inter_area = intersection_.area
        
        if intersection_.geom_type == 'Polygon':
            isect_x, isect_y = intersection_.exterior.coords.xy
            isect_.append({"X": list(isect_x), "Y": list(isect_y)})
            
        elif intersection_.geom_type == 'MultiPolygon':
            for part_ in intersection_:
                isect_x, isect_y = part_.exterior.coords.xy
                isect_.append({"X": list(isect_x), "Y": list(isect_y)})   
    else:
        inter_area = 0
        isect_x = -1
        isect_y = -1

    '''
        The commented section below could plot the annotation and patch box 
        at the same time. However, this process may slow down the whole pipeline.
        
    '''
        
    if not MultiPolygon_flag:
        if draw:
            fig = plt.figure(1, figsize=(7,7))
            ax = fig.add_subplot(111)
            ax.plot(x_, y_)

        bound_width = abs(poly_.bounds[2] - poly_.bounds[0])
        bound_height = abs(poly_.bounds[3] - poly_.bounds[1])

        if draw:
            ax.add_patch(
            patches.Rectangle(
                (poly_.bounds[0], poly_.bounds[1]),              # (x,y)
                bound_width,          # width
                bound_height,          # height
                fill=False,
                linestyle='dashdot'
                )
            )

            ax.add_patch(
            patches.Rectangle(
                (start_x, start_y),     # (x,y)
                patch_width,              # width
                patch_height,
                fill=False,
                edgecolor="yellow"
                )
            )

            # if not isect_x == -1:
            if intersection_.geom_type == 'MultiPolygon':
                for part_ in intersection_:
                    isect_x, isect_y = part_.exterior.coords.xy
                    ax.plot(isect_x, isect_y, color='tab:orange')

            elif intersection_.geom_type == 'Polygon':
                ax.plot(isect_x, isect_y, color='tab:orange')

            plt.axis('scaled')
            plt.show()
    
    return inter_area, isect_


def calculate_intersection(poly_, start_x, start_y, patch_width=500, patch_height=500, draw=False):
    
    """Calculates the whole intersection area (patch ∩ tumor area).
    
    The reason why 'calculate_polygon' exists is that the intersection could be one polygon
    area or multiple polygon areas, and we need to treat them seperately, though the latter
    circumstance is very rare.
    
    The commented section below could plot the annotation and patch box at the same time. 
    However, this process may slow down the whole pipeline.
    
    Args:
        poly_: a shapely Polygon object.
        
        start_x: min(patch.coordinates.x). (start_x, start_y) is the left-bottom point.
        start_x: min(patch.coordinates.y). 
        
        patch_width: width of the patch.
        patch_height: height of the patch.
    
    Returns:
        the size and coordinates of the intersection area (patch ∩ tumor area).
        Returns inter_area = 0, isect_x = -1, isect_y = -1 when (patch ∩ tumor area) == None. 
    
    Raises:
        None.
    """    
    
    MultiPolygon_flag = False
    inter_area = 0
    isect_ = []
    
    if not poly_.is_valid:
        poly_ = poly_.buffer(0)
    
    if poly_.geom_type == "MultiPolygon":
        
        MultiPolygon_flag = True
        
        multi_list_ = []

        if draw:       
            fig = plt.figure(2, figsize=(7,7))
            ax_ = fig.add_subplot(111)
            ax_.add_patch(
            patches.Rectangle(
                (start_x, start_y),     # (x,y)
                patch_width,              # width
                patch_height,
                fill=False,
                edgecolor="yellow"
                )
            )
        
        bound_x_min = 0
        bound_y_min = 0
        bound_x_max = 0
        bound_y_max = 0
        
        for poly_i in poly_:
            
            inter_area_i, isect_i = calculate_polygon(poly_i, start_x, start_y, 
                                                      patch_width, patch_height, MultiPolygon_flag)
            
            if not bound_x_min:
                bound_x_min = poly_i.bounds[0]
            else:
                if bound_x_min > poly_i.bounds[0]:
                    bound_x_min = poly_i.bounds[0]
            
            if not bound_y_min:
                bound_y_min = poly_i.bounds[1]
            else:
                if bound_y_min > poly_i.bounds[1]:
                    bound_y_min = poly_i.bounds[1]
            
            if not bound_x_max:
                bound_x_max = poly_i.bounds[2]
            else:
                if bound_x_max < poly_i.bounds[2]:
                    bound_x_max = poly_i.bounds[2]
            
            if not bound_y_max:
                bound_y_max = poly_i.bounds[3]
            else:
                if bound_y_max < poly_i.bounds[3]:
                    bound_y_max = poly_i.bounds[3]
            
            if draw:    
                x_, y_ = poly_i.exterior.coords.xy
                ax_.plot(x_, y_, color='tab:blue')
            
            inter_area += inter_area_i
            
            for isect_part in isect_i:
                isect_.append(isect_part)

        bound_width = abs(bound_x_max - bound_x_min)
        bound_height = abs(bound_y_max - bound_y_min)

        if draw:        
            for _ in isect_:
                ax_.plot(_['X'], _['Y'], color='tab:orange')
            
            ax_.add_patch(
            patches.Rectangle(
                (bound_x_min, bound_y_min),              # (x,y)
                bound_width,          # width
                bound_height,          # height
                fill=False,
                linestyle='dashdot'
                )
            )
            
            plt.axis('scaled')
            plt.show()
        
    elif poly_.geom_type == "Polygon":
        inter_area, isect_ = calculate_polygon(poly_, start_x, start_y, 
                                               patch_width, patch_height, MultiPolygon_flag)
        
    return inter_area, isect_

'''
    Calculate tumor area.
'''
def calc_tumorArea(polygon_list, patches_coords):
    '''
    '''
    patch_size=500

    area_list = dict()

    for coords in patches_coords:
    
        x_, y_ = coords
        
        area_sum = 0

        for idx_, poly_ in enumerate(polygon_list):

            area_, _ = calculate_intersection(poly_, x_, y_)
            area_sum += area_
            # print((x_, y_), area_)

        if int(area_sum) > 0:
            print((x_, y_), "sum:", area_sum / (patch_size * patch_size), area_sum)

            if area_sum >= patch_size * patch_size:
                area_sum = patch_size * patch_size

            area_list[coords] = int(area_sum)

        else:
            area_list[coords] = 0

    return area_list

def preprocessingAndanalysis(slide_name, section_list, positivethresh, \
    dataset_dir='./dataset_patches/', level_dir='/level1/'):
    
    '''
    Args:
        - slide_name: for example, 'patient_015_node_2',
        In current case, DO NOT add '.tif' in slide_name;
        
        - section_list: the sections to be analyzed. For example:
            
        section_list = ['00', '01', '02', '03', \
                        '10', '11', '12', '13', \
                        '20', '21', '22', '23', \
                        '30', '31', '32', '33']        

        - dataset_dir: directory to datasets;

        - positivethresh: discard patches in which tumor area is too small;

    '''
    
    '''
        Section1: locate all patches and coordinates dirs.
    '''
    patches_dir_all = list()
    coordinates_file_all = list()

    for sect in section_list:

        dir_ = dataset_dir + slide_name + level_dir + sect
        patches_dir = dataset_dir + slide_name + level_dir + sect + '/patches/'
        if not os.path.isdir(dir_):
            continue

        coordinates_file = [i for i in os.listdir(dir_) \
                            if '.csv' in i][0]
        coordxml_file = dataset_dir + slide_name + level_dir + sect + '/' + coordinates_file
        # print(patches_dir)
        patches_dir_all.append(patches_dir)
        # print(coordxml_file)
        coordinates_file_all.append(coordxml_file)
    
    '''
        Section2: collect paths to all the patch images.
    '''
    instances_all = list()

    for patch_dir_ in patches_dir_all:
        tmp = [patch_dir_ + i for i in os.listdir(patch_dir_) if '.jpeg' in i]
        instances_all += tmp

    assert len(list(set(instances_all))) == len(instances_all)
    print(slide_name)
    print("total number of patches: ", len(instances_all))
    
    
    '''
        Section3: re-organize coordinates with pandas DataFrame. 
    '''
    frames = list()
    for coordsfile in coordinates_file_all:

        df_tmp = pd.read_csv(coordsfile)
        frames.append(df_tmp)
    
    # pd_all: DataFrame which holds all the coords
    pd_all = pd.concat(frames)
    pd_all = pd_all.drop_duplicates()
    
    # pd_dist: DF in which rows were ordered by column 'tumor area'.
    pd_dist = pd_all.groupby('tumor_area').size()\
              .reset_index().rename(columns={0:'numbers'})
    
    plt.hist(pd_dist.tumor_area, weights=pd_dist.numbers, align='mid')

    plt.xlabel('Area Size')
    plt.ylabel('Number')
    plt.title('Tumor patch statistics of' + ' ' +  slide_name)
    plt.show()
    
    # DF which holds only tumor patches

    positivethresh = float(positivethresh)

    pd_tumor = pd_all.loc[pd_all['tumor_area'] > 0].\
            sort_values(by=['tumor_area'], ascending=False).reset_index()

    pd_valid_tumor = pd_all.loc[pd_all['tumor_%'] > positivethresh].\
            sort_values(by=['tumor_area'], ascending=False).reset_index()

    print("Example of tumor patches:\n", pd_tumor[:10])

    print("Example of valid tumor patches:\n", pd_valid_tumor[:10])
    
    '''
        Section4: Save paths of postive and negative seperately
    '''
    tumor_coords = list()

    for index, row in pd_valid_tumor.iterrows():
        x_ = int(row['coord_x'])
        y_ = int(row['coord_y'])
        tumor_coords.append((x_, y_)) 
        # print(x_, y_) 

    positive_patches_path = list()
    negative_patches_path = list()

    for patch_ in instances_all:

        filename = patch_.split('/')[-1].split('.')[0]
        x_ = int(filename.split('_')[-2])
        y_ = int(filename.split('_')[-1])

        if (x_, y_) in tumor_coords:
            positive_patches_path.append(patch_)
        else:
            negative_patches_path.append(patch_)

    assert (len(negative_patches_path) + len(positive_patches_path)) == len(instances_all)
    
    print("Number of positive patches (valid): ", len(positive_patches_path))
    print("Number of negative patches: ", len(negative_patches_path))
    
    # change this to the target dir as desired
    cur_dir=dataset_dir + slide_name + level_dir

    if len(positive_patches_path) != 0:
        
        with open(cur_dir + 'pospaths.txt', "wb") as f:   
             pickle.dump(positive_patches_path, f)

    with open(cur_dir + 'negpaths.txt', "wb") as f:   
         pickle.dump(negative_patches_path, f)
    
    return pd_all, pd_tumor, positive_patches_path, negative_patches_path


'''
    Draw extracted patches.
'''

def draw_pospatch(patchpath, slidepath, annopath, level, \
    mag_factor,delta_x=0, delta_y=0):
    
    '''
    
    '''
    
    PIXEL_BLACK = 0
    PIXEL_WHITE = 255
    patch_size=500
    
    samplepos = Image.open(patchpath)

    if delta_x ==0 and delta_y==0:
        delta_x=int(patchpath.split('/')[-1].split('.')[0].split('_')[-2])
        delta_y=int(patchpath.split('/')[-1].split('.')[0].split('_')[-1])

    print("delta x y", delta_x, delta_y)

    polygon_list, anno_list = parse_annotation(annopath, level, mag_factor)
    local_anno = list()

    for area_ in anno_list:
        tmp = list()
        for coords in area_:

            x_ = coords[0] - delta_x
            y_ = coords[1] - delta_y

            # if x_ < 0:
            #     x_ = 0
            # if y_ < 0:
            #     y_ = 0

            # if x_ > patch_size:
            #     x_ = patch_size
            # if y_ > patch_size:
            #     y_ = patch_size

            tmp.append((x_, y_))
        local_anno.append(tmp)

    local_anno_arr = list()
    for contour in local_anno:

        arr = np.array(contour)
        arr = np.expand_dims(arr, axis=1)
        local_anno_arr.append(np.array(arr))

    sample_arr = np.array(samplepos) 
    sample_filled=cv2.drawContours(sample_arr, local_anno_arr, -1, 
                                  (0, 255, 0, 255),thickness=-1)
    
    patchfilled_img = Image.fromarray(sample_filled)
    
    sample_arr = np.array(samplepos) 
    sample_annotated=cv2.drawContours(sample_arr, local_anno_arr, -1, 
                                     (0, 255, 0, 255),thickness=3)

    patchannotated_img = Image.fromarray(sample_annotated)

    return patchfilled_img, patchannotated_img


'''
    The whole pipeline of extracting patches.
'''
def extract_all(slide_path, anno_path, level, mag_factor, pnflag=True):
    '''
    Args:
        slide_path: Path to target slide, 
            for example: ../data-wsi/camelyon17/training/patient_017_node_2.tif
    
        anno_path: Path to the annotation xml of the target slide, 
            for example: ../data-wsi/camelyon17/lesion_annotations/patient_017_node_2.xml
    
        level: Magnification level; 

        mag_factor: Pow(2, level);
        
        pnflag: Boolean, which indicates whether it is a positive one or not
    
    Returns: 
        To-do.
    '''
    
    section_list = ['00', '01', '02', '03', \
                    '10', '11', '12', '13', \
                    '20', '21', '22', '23', \
                    '30', '31', '32', '33']

    patches_all = list()

    wsi_obj=openSlide_init(slide_path, level)

    if pnflag:
        polygon_list, anno_list = parse_annotation(anno_path, level, mag_factor)

    time_all = 0

    for sect in section_list:
        
        start = time.time()

        rgba_image = read_wsi(wsi_obj, level, mag_factor, sect)
        wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_image)
        # print('Transformed shape: (height, width, channel)')
        # print("WSI HSV shape: ", wsi_hsv_.shape)
        # print("WSI RGB shape: ", wsi_rgb_.shape)
        # print("WSI GRAY shape: ", wsi_gray_.shape)
        # print('\n')

        del rgba_image
        gc.collect()

        bounding_boxes, contour_coords, contours, mask \
        = segmentation_hsv(wsi_hsv_, wsi_rgb_)

        del wsi_hsv_
        gc.collect()

        patches, patches_coords, patches_coords_local\
        = construct_bags(wsi_obj, wsi_rgb_, contours, mask, \
                        level, mag_factor, PATCH_SIZE, sect)
        
        if len(patches):
            patches_all.append(patches)
            if pnflag:
                tumor_dict = calc_tumorArea(polygon_list, patches_coords)
            else:
                tumor_dict = None

            save_to_disk(patches, patches_coords, tumor_dict, mask, \
                         slide_path, level, sect)

        del wsi_rgb_
        del patches
        del mask
        gc.collect()
        
        end = time.time()
        time_all += end - start
        print("Time spent on section", sect,  (end - start), '\n')
    
    print('total time: ', time_all)    
    
    return patches_all

def extract_all_Plus(slide_path, anno_path, section_list, pnflag=True, level=1):

    '''
    Args:
        - slide_path: Path to target slide, 
            for example: ../data-wsi/camelyon17/training/patient_017_node_2.tif
    
        - anno_path: Path to the annotation xml of the target slide, 
            for example: ../data-wsi/camelyon17/lesion_annotations/patient_017_node_2.xml
    
        - level: Magnification level;
        
        - pnflag: Boolean, which indicates whether it is a positive one or not
    
    Returns: 
        To-do.
    '''
    
    mag_factor = pow(2, level)
    
    slide_name = slide_path.split('/')[-1].split('.')[0]
    
    wsi_obj = openSlide_init(slide_path, level)

    if pnflag:
        '''
            if this slide is an annotated positive slide.
        '''
        polygon_list, anno_list = parse_annotation(anno_path, level, mag_factor)

    time_all = 0

    patches_all = list()

    for sect in section_list:

        start = time.time()

        rgba_image = read_wsi(wsi_obj, level, mag_factor, sect)
        wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_image)

        del rgba_image
        gc.collect()

        bounding_boxes, contour_coords, contours, mask \
        = segmentation_hsv(wsi_hsv_, wsi_rgb_)

        del wsi_hsv_
        del wsi_gray_
        gc.collect()

        patches, patches_coords, patches_coords_local\
        = construct_bags(wsi_obj, wsi_rgb_, contours, mask, \
                        level, mag_factor, sect, PATCH_SIZE, SPLIT)

        if len(patches):
            patches_all.append(patches)
            if pnflag:
                tumor_dict = calc_tumorArea(polygon_list, patches_coords)
            else:
                tumor_dict = None

            save_to_disk(patches, patches_coords, tumor_dict, mask, \
                         slide_path, level, sect)

        del wsi_rgb_
        del patches
        del mask
        gc.collect()

        end = time.time()
        time_all += end - start
        print("Time spent on section", sect,  (end - start))
            
    samplelevel=7
    samplemag_factor=pow(2, samplelevel)

    print(slide_path)
    print(wsi_obj.level_dimensions[samplelevel])

    width_thumb, height_thumb = wsi_obj.level_dimensions[samplelevel]

    wsi_img = wsi_obj.read_region((0, 0), samplelevel, (width_thumb, height_thumb))
    wsi_arr = np.array(wsi_img)
    
    if pnflag:
        tree = ET.ElementTree(file=anno_path)

        anno_ = list()

        for an_i, crds in enumerate(tree.iter(tag='Coordinates')):
            tmp = list()
            for coord in crds:
                x = int(float(coord.attrib['X'])) // samplemag_factor
                y = int(float(coord.attrib['Y'])) // samplemag_factor
                tmp.append((x, y))

            anno_.append(tmp)

        anno_arr = list()

        for contour in anno_:

            arr = np.array(contour)
            arr = np.expand_dims(arr, axis=1)
            anno_arr.append(np.array(arr))

        _=cv2.drawContours(wsi_arr, anno_arr, -1, \
                          (0, 255, 0, 255),thickness=2)

    img_sample_ = Image.fromarray(wsi_arr[:,:,:3])
    
    print('total time: ', time_all)

    return img_sample_, wsi_img

'''
    Back up function for multiprocessing.

    Here is a simple example of patch extraction using the extract_all_Plus function above:

    slidelist= ['patient_019_node_2','patient_019_node_3','patient_020_node_1',\
                'patient_021_node_1','patient_021_node_4','patient_022_node_0']
    p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-2)
    p.map(capextract, slidelist)
    p.close()
    p.join()

'''
def capextract(slide_name, dataset_dir='./dataset_patches/', level_dir='/level1/'):

    pnflag=False

    training_dir='../data-wsi/camelyon17/training/'
    anno_dir = '../data-wsi/camelyon17/lesion_annotations_new/'

    dir_prefix='./dataset_patches/' 
    dir_end='/level1/patches/'

    slideEnd='.tif'
    annoEnd='.xml'

    slide_path = training_dir + slide_name + slideEnd
    anno_path = anno_dir + slide_name + annoEnd

    print("SSS: ", slide_path)
    
    try:
        section_list0 = ['00', '01', '02', '03']
        img_sample_, wsi_img = extract_all_Plus(slide_path, anno_path, section_list0, pnflag, level=1)
        del img_sample_
        del wsi_img
        gc.collect()
        
        section_list1 = ['10', '11', '12', '13']
        img_sample_, wsi_img = extract_all_Plus(slide_path, anno_path, section_list1, pnflag, level=1)
        del img_sample_
        del wsi_img
        gc.collect()

        print('sec list 2\n\n')
        section_list2 = ['20', '21', '22', '23']
        img_sample_, wsi_img = extract_all_Plus(slide_path, anno_path, section_list2, pnflag, level=1)
        del img_sample_
        del wsi_img
        gc.collect()

        print('sec list 3\n\n')    
        section_list3 = ['30', '31', '32', '33']
        img_sample_, wsi_img = extract_all_Plus(slide_path, anno_path, section_list3, pnflag, level=1)
        del img_sample_
        del wsi_img
        gc.collect()  

    except:
        print("???:", slide_name)

    else:

        section_list = ['00', '01', '02', '03', \
                        '10', '11', '12', '13', \
                        '20', '21', '22', '23', \
                        '30', '31', '32', '33']  

        pd_all, pd_tumor, positive_patches_path, negative_patches_path = \
        preprocessingAndanalysis(slide_name, section_list, dataset_dir, level_dir, \
                                 positivethresh=0)

