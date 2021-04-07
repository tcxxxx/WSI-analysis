## Patch Extraction / Dataset Praparation <br>
Patch extraction is the most important preparatory work when we adopt patch-based methods to analyze WSIs. However, due to the variance among WSIs, this procedure could be rather difficult.<br>

Currently, there are three files in the directory:
- extract_patches.py: includes processing functions for WSIs of which **level >= 3**;
- extract_patches_split.py: includes processing functions for WSIs of which **level <= 2**. The solution we currently adopt is to split the whole WSI images first, and process the sections in turn;<br>
~~- utils.py: includes functions for tumor area calculation;<br>~~

**It should be noticed that these files include seperate functions only and they are not the complete script to process WSIs. All the functions need to be imported before usage:<br>**
``````
from WSIAnalysis.patch_extraction.extract_patches_split import \
openSlide_init, read_wsi, construct_colored_wsi, get_contours, \
segmentation_hsv, construct_bags, save_to_disk, parse_annotation, \
preprocessingAndanalysis, draw_pospatch, calculate_intersection, \
calculate_polygon, calc_tumorArea, \
extract_all_Plus
``````

For convenience, all the functions are written in the *extract_patches_split.py* file. Please do not hesitate to leave a comment if you have any advice or run into problems with these codes. :)

----------------------------------------------------------------------------
### Examples:
- [x] [HERE](./examples/patch_extraction_level3example.html) (better when downloaded then viewed using a browser) is an example of level**3** patch-extraction pipeline. Jupyter notebook file is also available in this directory. 
- [x] [HERE](./examples/patch_extraction_level1(20180419).html) (better when downloaded then viewed using a browser) is an example of level**1** patch-extraction pipeline (updated on 2018.04.19). <br> It should be noticed that the functions of processing level**1** WSI (in **extract_patches_split.py**) are somewhat different from the ones in **extract_patches.py**, which were designed for images whose level >= 3.

---------------------------------------------------------------------------
There are several tricky parts when extracting patches from WSIs:
1. **Memory limit.** <br>
The RAM size of our lab is 31 GB, and it could hardly hold a level0 WSI. So be careful when loading the whole image.<br>
It is also helpful to use **del** and **gc.collect()** to free up memory.
2. **Coordinates scaling level/reference frame.** <br>
The read_region() method in [OpenSlide](http://openslide.org/api/python/) processes WSIs in level 0 reference frame. So
necessary transformation is needed when we crop patches from WSIs using read_region() method.
3. **Shape difference between Pillow Image object and NumPy arrays.** <br>
numpy.asarray() / numpy.array() would switch the position of WIDTH and HEIGHT in shape, and vice versa. 
If an Image object' shape is (WIDTH, HEIGHT, CHANNEL), the shape will be (HEIGHT, WIDTH, CHANNEL) after the np.asarray() transformation.
4. **Magnification level choice**<br>
Below is an patch-extraction example (performed on one sample from [Camelyon 2017 dataset](https://camelyon17.grand-challenge.org/data/)). Red boxes are selected patches and green ones annotated tumor areas. As we can see, when we extract 500 x 500 patches from a WSI in **level3** scale, the portion of tumor areas are too small, which means discriminative information could be significantly diluted if we use all these selected patches to train CNN. <br>This urges us to use smaller magnification level (higher resolution scale).
![slide09-1](http://119.29.151.114/images/level3_patche_extraction.jpeg)

----------------------------------------------------------------------------
### References / Helpful links:
1. [Drawing contours with OpenCV Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html)
2. [Camelyon 2017 Grand Challenge](https://camelyon17.grand-challenge.org/)
3. [Camelyon 2016 Grand Challenge](https://camelyon16.grand-challenge.org/)
4. [State-of-the-art patch-based strategy for WSI analysis](https://arxiv.org/pdf/1504.07947.pdf)
