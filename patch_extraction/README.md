## Patch Extraction / Dataset Praparation <br>
Patch extraction is the most important preparatory work when we adopt patch-based methods to analyze WSIs. However, due to the variance among WSIs, this procedure could be rather difficult.<br>

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

In the **html** and **notebook** files of this directory is an example of the patch-extraction pipeline.
