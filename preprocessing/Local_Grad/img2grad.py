# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:12:46 2024

@author: danhv
"""

import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import util
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm


parser = argparse.ArgumentParser(description='Local Gradient image processing')
parser.add_argument('--input_dir', default='resources',
    type=str, help='Directory of images need to convert to gradient')
parser.add_argument('--result_dir', default='result',
    type=str, help='Directory for results')

args = parser.parse_args()


#util.mkdir(args.result_dir)


processimg = transforms.Compose([
            transforms.ToTensor(),

        ])

KERNEL_X = np.array([[0, 0, 0],
                  [0, -1, 1],
                  [0, 0, 0]])
KERNEL_Y = np.array([[0, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0]])


def cv2_read_image(image_path):
    image = cv2.imread(image_path)
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    chs = cv2.split(image)
    return cv2.merge((chs[0], chs[1], chs[2]))


def get_image_paths(based_dir):
    
    image_paths = []
    
    for dirpath, dirnames, filenames in os.walk(based_dir):
        if len(filenames)>0:
            imgs = [os.path.join(dirpath, f) for f in filenames \
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.webp')) ]
            
            image_paths.extend(imgs)
        
    return image_paths

def grad_process(image,kernel_x = KERNEL_X,kernel_y=KERNEL_Y):
    
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    b = grad_magnitude[:,:,0] * 255 / grad_magnitude[:,:,0].max()
    g = grad_magnitude[:,:,1] * 255 / grad_magnitude[:,:,0].max()
    r = grad_magnitude[:,:,2] * 255 / grad_magnitude[:,:,0].max()
    
    return cv2.merge((r,g,b)).astype(np.uint8)

def save_image(filename, image):
    cv2.imwrite(filename, image)
    

def show_img(image:torch.Tensor, cmap='gray'):
    if type(image) is torch.Tensor:
        plt.imshow(image.permute(1, 2, 0), cmap='gray')
        return
    
    try:
        plt.imshow(image, cmap='gray')
    except:
        return
    
if __name__ == '__main__':
    
    args.input_dir = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
    args.result_dir = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset_grad'
    image_paths = get_image_paths(args.input_dir)
    
    for path in tqdm(image_paths):
        img = cv2_read_image(path)
        image = grad_process(img)
        
        rel_path = os.path.relpath(path, args.input_dir)
        
        dst_path = os.path.join(args.result_dir, rel_path)
        
        util.mkdir(os.path.dirname(dst_path))
        
        save_image(dst_path, image)
        
    
    
    
    
    


# =============================================================================
# # Define the directory path
# dir_path = "/home/user/documents"
# 
# # Use os.walk to iterate over the directory tree
# for dirpath, dirnames, filenames in os.walk(dir_path):
#     print("Current directory:", dirpath)
#     print("Subdirectories:", dirnames)
#     print("Files:", filenames)
#     print()
# =============================================================================
    
    
    
    
    
    