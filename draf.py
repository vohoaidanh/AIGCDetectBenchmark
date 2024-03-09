import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_gaussian_angle_map(image, sigma):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradient using Sobel operators
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and angle
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    
    # Apply Gaussian smoothing to the angle map
    angle_map = cv2.GaussianBlur(angle, (0, 0), sigma)
    angle_map = angle
    return angle_map

# Read an image
image = cv2.imread('resources/000609612.jpg')

D = 224
r = D / image.shape[0]
dim = (int(image.shape[1] * r), D)
image = cv2.resize(image, dim)
# Calculate Gaussian angle map with sigma = 1
gaussian_angle_map = calculate_gaussian_angle_map(image, 1)

plt.imshow(gaussian_angle_map*255.0/2/np.pi, cmap='gray')
plt.show()
plt.imshow(image[:,:,:], cmap='gray')
plt.show()


import os
import csv
import torch

from validate import validate,validate_single
from options import TestOptions
from eval_config import *
from PIL import ImageFile
from util import create_argparser,get_model, set_random_seed
from data import create_dataloader, create_dataloader_new

ImageFile.LOAD_TRUNCATED_IMAGES = True


opt = TestOptions().parse(print_options=True) #获取参数类
opt.dataroot = '{}/{}'.format(dataroot, 'train')
opt.dataset_name = 'ELSA'
data_loader = create_dataloader_new(opt)






