# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:50:38 2022

@author: Shell Lin
"""

import cv2
import numpy as np
import os

########################## KEY PARAMETERS ################################

# Path
colourfile = r"D:\Jiejie stuff\CAM N-N\Hydrogel\p\4.tif"
filename = "img_mean (block=25, c=0)_colormask.tif"
filedir = r"D:\Jiejie stuff\Shell_Program\Download\4" 
filepath = filedir + '\\' + filename
downloadPath = r"D:\Jiejie stuff\Shell_Program\Download\4\Skeleton"


img_colour = cv2.imread(colourfile,0)
img_original = cv2.imread(filepath,0)
img_reverted =  cv2.bitwise_not(img_original) 
img = img_reverted

########################## SKELETON ################################

# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open_ = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, open_)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break


########################## DISPLAY & DOWNLOAD ################################

# Change the current directory 
# to specified directory 
os.chdir(downloadPath)

filename = filename[:-4] + '_skeleton.tif'
# =============================================================================
# filename_mean = 'img_mean (block=' + str(block) + ', c='+ str(c)+')_colormask.tif'
# filename_gaus = 'img_gaussian (block=' + str(block) + ', c='+ str(c)+')_colormask.tif'
# =============================================================================

# Saving the image
cv2.imwrite(filename, skel)


# Displaying the final skeleton

cv2.imshow("Original vs Skeleton", np.hstack([img_colour, img_original, skel]) )
cv2.waitKey(0)
cv2.destroyAllWindows()


