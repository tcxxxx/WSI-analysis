'''
    Written in Python 3.5
'''

import xml.etree.cElementTree as ET
import os
import os.path
import csv

# import openslide

import numpy as np
import pandas as pd
from shapely.geometry import box, Point, Polygon

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from ast import literal_eval

import pickle

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
    #, isect_x, isect_y

'''
    Calculate tumor area.
'''
def calc_tumorArea(polygon_list, patches_coords):
    '''
    '''
    area_list = dict()

    for coords in patches_coords:
    
        x_, y_ = coords
        
        area_sum = 0

        for idx_, poly_ in enumerate(polygon_list):

            area_, _ = calculate_intersection(poly_, x_, y_)
            area_sum += area_

        if int(area_sum) > 0:
            print((x_, y_), ":", area_sum / (500*500), area_sum)
            
            area_list[coords] = int(area_sum)

        else:
            area_list[coords] = 0

    return area_list


'''
    !!! Take care because the directory structure could be rather different.
    Here we suppose you extracted patches with functions from
    extract_patches.py / extract_patches_split.py. 
    
    The directory should be like:
    
    ./dataset_patches
    |
    |-- patient_015_node_2 (dir for each slide)
    |   |
    |   |-- level1 (dir for level 1)
    |   |   |
    |   |   |-- 01
    |   |   |    |--  mask.npy
    |   |   |    |--  patch_coords01.csv  
    |   |   |    |--  patch_whole01.npy 
    |   |   |    \--  patches 
    |   |   |         |-- (all the patches)   
    |   |   |
    |   |   |-- 02
    |   |   |    |-- ...    
    |   |   |-- 11
    |   |   |    |-- ...
    |   |   |-- 12
    |   |   |    |-- ...
    |   |   |-- 21
    |   |   |    |-- ...
    |   |   |-- 22  
    |   |   |    |-- ...
    
'''
def preprocessingAndanalysis(slide_name, section_list, positivethresh, \
	dataset_dir='./dataset_patches/', level_dir='/level1/'):
    
    '''
    Args:
        slide_name: for example, 'patient_015_node_2',
        In current case, DO NOT add '.tif' in slide_name;
        section_list: the sections to be analyzed;
        dataset_dir: dir / path;
        positivethresh: discard patches in which tumor area is too small.

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

    # with open(cur_dir + 'pospaths.txt', "rb") as fp:   # Unpickling
    #     b = pickle.load(fp)
    
    return pd_all, pd_tumor, positive_patches_path, negative_patches_path
    
# pd_all, pd_tumor, positive_patches_path, negative_patches_path = \
# pre_analysis(slide_)   


