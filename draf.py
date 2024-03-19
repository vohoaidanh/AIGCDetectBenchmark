import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader_new, create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options import TrainOptions
from data.process import get_processing_model
from util import set_random_seed


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt





 
from PIL import Image
import os

def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths


class read_data_new():
    def __init__(self, dataroot):
        self.root = dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]

        return img, target

    def __len__(self):
        return len(self.label)


dataroot = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'

dset = read_data_new(dataroot)

dataiter = iter(dset)

smaple = next(dataiter)

a = smaple[1]






import os
import shutil

def copy_images(source_folder, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print("Source folder does not exist.")
        return
    
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Recursively copy the entire directory tree
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # Check if the file is an image (you can customize this condition)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                source_path = os.path.join(root, filename)
                relative_path = os.path.relpath(source_path, source_folder)
                destination_path = os.path.join(destination_folder, relative_path)
                destination_dir = os.path.dirname(destination_path)
                # Create the directory if it doesn't exist
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                # Copy the image file to the destination folder
                shutil.copyfile(source_path, destination_path)
                print(f"Copied {filename} to {destination_folder}")


#source_folder = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset'
#destination_folder = r'D:\K32\do_an_tot_nghiep\data\real_gen_dataset2'
#copy_images(source_folder, destination_folder)


import numpy as np
np.random.choice()




input_tensor = torch.randn(1, 2048)  # Assuming batch size is 1














