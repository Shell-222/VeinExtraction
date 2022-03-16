# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:03:51 2022

@author: Shell Lin
"""

import cv2 as cv
import numpy as np
import os
########################## KEY PARAMETERS ################################

# Path
filepath = r"D:\Jiejie stuff\CAM N-N\Hydrogel\p\3.tif"
downloadPath = r"D:\Jiejie stuff\Shell_Program\Download\3"

# Greyscale Threshold (default 127)
thres = 127 

# Block size (default 11): 
# Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
block = 25

# C constant (default 2):
# Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
c = 0

# Original image (colour)
img_origin = cv.imread(filepath, cv.IMREAD_COLOR)


########################## COLOUR MASK ################################
# define the limit of boundaries ()
lower =   [20, 15, 110] # [17, 15, 90]
upper =   [204, 204, 255]  # [204, 204, 255]  


# create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

# find the colors within the specified boundaries and apply the mask
mask = cv.inRange(img_origin, lower, upper)
img_colormask = cv.bitwise_and(img_origin, img_origin, mask = mask)

# =============================================================================
# # show the images
# cv.imshow("images", np.hstack([img_origin, img_colormask]))
# cv.waitKey(0)
# =============================================================================


########################## THRESHOLDING ################################


# Greyscale image
img_greyscale = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
# =============================================================================
# img_greyscale = cv.cvtColor(img_colormask, cv.COLOR_BGR2GRAY)
# =============================================================================

# I don't knwo why we need to blur the image
img_greyscale = cv.medianBlur(img_greyscale,5)

# Simple Thresholding
ret,img_simple = cv.threshold(img_greyscale,thres,255,cv.THRESH_BINARY)

# Adaptive Thresholding: The threshold value is the mean of the neighbourhood area minus the constant C.
img_mean = cv.adaptiveThreshold(img_greyscale,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,block,c)

# Adaptive Thresholding: The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.
img_gaussian = cv.adaptiveThreshold(img_greyscale,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,block,c)




########################## SKELETON ################################

img_reverted =  cv.bitwise_not(img_mean) # Use mean image for skeleton


# Step 1: Create an empty skeleton
size = np.size(img_reverted)
img_skel = np.zeros(img_reverted.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open_ = cv.morphologyEx(img_reverted, cv.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv.subtract(img_reverted, open_)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv.erode(img_reverted, element)
    img_skel = cv.bitwise_or(img_skel,temp)
    img_reverted = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv.countNonZero(img_reverted)==0:
        break




########################## DISPLAY & DOWNLOAD ################################
# Change the current directory 
# to specified directory 
os.chdir(downloadPath)

filename_mean = 'img_mean (block=' + str(block) + ', c='+ str(c)+').tif'
filename_gaus = 'img_gaussian (block=' + str(block) + ', c='+ str(c)+').tif'
filename_skel = 'img_mean (block=' + str(block) + ', c='+ str(c)+')_skeleton.tif'
# =============================================================================
# filename_mean = 'img_mean (block=' + str(block) + ', c='+ str(c)+')_colormask.tif'
# filename_gaus = 'img_gaussian (block=' + str(block) + ', c='+ str(c)+')_colormask.tif'
# =============================================================================

# Saving the image
cv.imwrite(filename_mean, img_mean)
cv.imwrite(filename_gaus, img_gaussian)
cv.imwrite(filename_skel, img_skel)


# Displaying the images
cv.imshow('img_origin', img_origin)
#cv.imshow("image_colormasked",  img_colormask)
#cv.imshow('img_simple (v=' + str(thres) + ')', img_simple)
cv.imshow('img_mean (block=' + str(block) + ', c='+ str(c)+')', img_mean)
cv.imshow('img_gaussian (block=' + str(block) + ', c='+ str(c)+')', img_gaussian)




cv.waitKey(0)
cv.destroyAllWindows()

