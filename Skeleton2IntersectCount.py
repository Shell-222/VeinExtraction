# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:51:33 2022

@author: Shell Lin
"""

import cv2
import numpy as np


#1. Create a binary image that has same size as input image, with a circle in ceter. 
#2. Somehow compute the intersection count of the circle and skeleton.

filename = "img_mean (block=25, c=0)_skeleton.tif"
filedir = r"D:\Jiejie stuff\Shell_Program\Download\4\Skeleton" 
filepath = filedir + '\\' + filename

# Load the skeleton image
img_skeleton = cv2.imread(filepath, 0);

row = len(img_skeleton)
colum = len(img_skeleton)

########### Use test skeleton to verify the count number for now
test_skeleton = [[0 for x in range(colum)] for y in range(row)]
test_skeleton = np.array(test_skeleton, dtype = "uint8")
# Draw a few lines on the test skeleton
test_skeleton[100][:] = 255
test_skeleton[200][:] = 255
test_skeleton[300][:] = 255
test_skeleton[400][:] = 255


# Center coordinates
center_coordinates = (int(row/2), int(colum/2))
 
# Radius of circle
radius = 90
  
# while color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1
  
# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(test_skeleton, center_coordinates, radius, color, thickness)
  
# Displaying the image
cv2.imshow("Skeleton with circle", image)



cv2.waitKey(0)
cv2.destroyAllWindows()