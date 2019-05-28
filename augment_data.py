
#Import necessary modules
from scipy import misc, ndimage
import os
import sys

#Load images from the images dir
IMG_DIR = sys.argv[1]
AUG_DIR = sys.argv[1]

for img in os.listdir(IMG_DIR):
    aug_name = img.split(".", 1)
    if(aug_name[1] == "jpg"):
        ori_img = misc.imread(os.path.join(IMG_DIR, img))
        ori_img = ndimage.gaussian_filter(ori_img, sigma=2)
        dest_name = aug_name[0] + "gb." + aug_name[1]
        misc.imsave( os.path.join(AUG_DIR, dest_name) , ori_img)