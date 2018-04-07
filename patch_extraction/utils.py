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

def calculate_polygon(poly_, start_x, start_y, patch_width, 
                      patch_height, MultiPolygon_flag):
    
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
            # ax.plot(isect_x, isect_y, linestyle='dashed', color = 'r')
            isect_.append({"X": list(isect_x), "Y": list(isect_y)})
            
        elif intersection_.geom_type == 'MultiPolygon':
            for part_ in intersection_:
                isect_x, isect_y = part_.exterior.coords.xy
                isect_.append({"X": list(isect_x), "Y": list(isect_y)})
                # ax.plot(isect_x, isect_y, linestyle='dashed', color = 'r')
    
    else:
        inter_area = 0
        isect_x = -1
        isect_y = -1

    '''
        The commented section below could plot the annotation and patch box 
        at the same time. However, this process may slow down the whole pipeline.
        
    '''
        
    if not MultiPolygon_flag:
        
        fig = plt.figure(1, figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.plot(x_, y_)

        bound_width = abs(poly_.bounds[2] - poly_.bounds[0])
        bound_height = abs(poly_.bounds[3] - poly_.bounds[1])

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

def calculate_intersection(poly_, start_x, start_y, patch_width=1024, patch_height=1024):
    
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

        #       
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
                
            x_, y_ = poly_i.exterior.coords.xy
            ax_.plot(x_, y_, color='tab:blue')
            
            inter_area += inter_area_i
            
            for isect_part in isect_i:
                isect_.append(isect_part)
        
        for _ in isect_:
            ax_.plot(_['X'], _['Y'], color='tab:orange')
        
        bound_width = abs(bound_x_max - bound_x_min)
        bound_height = abs(bound_y_max - bound_y_min)
        
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