import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
# Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class SatelliteDataset(Dataset):
    def __init__(self, data, patch_size=256, tile_size = 500, transform=None):
        self.data = data
        self.transform = transform
        self.sat_img_set = glob.glob(os.path.join(self.data, "images/*.tif"))
        self.sat_img_set.sort()
        self.gt_img_set = glob.glob(os.path.join(self.data, "gt/*.tif"))
        self.gt_img_set.sort()
        self.tile_size = tile_size
        self.patch_size = patch_size
        # TODO - as the input image resolution is  5000*5000, this is hardcoded.
        self.n_rows = 5000 // self.tile_size
        self.n_columns = 5000 // self.tile_size
        self.n_of_patches = self.n_rows * self.n_columns
    
    def __len__(self):
      # single image is now tiled. So total number of patches is number of patches times the available images.
        return len(self.sat_img_set) * self.n_of_patches
    
    def __getitem__(self, index):
        # image id to refer.
        image_id = index // self.n_of_patches
        # patch to refer in the identified image
        patch_id = image_id % self.n_of_patches
        # spatial location of the patch in the image
        patch_row_id = patch_id // self.n_columns
        patch_col_id = patch_id % self.n_columns
        sat_img = cv2.imread(self.sat_img_set[image_id])
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.imread(self.gt_img_set[image_id])
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        # get the required patch from the image
        sat_patch = sat_img[patch_row_id*self.tile_size:(patch_row_id+1)*self.tile_size, patch_col_id*self.tile_size:(patch_col_id+1)*self.tile_size]
        gt_patch = gt_img[patch_row_id*self.tile_size:(patch_row_id+1)*self.tile_size, patch_col_id*self.tile_size:(patch_col_id+1)*self.tile_size]
            
        if self.transform:
            transformed = self.transform(self.patch_size, image=sat_patch, mask=gt_patch)
            sat_img = transformed['image']
            gt_img = transformed['mask']
            # add channel dimension to mask images
            gt_img = torch.unsqueeze(gt_img, 0)
        
        img_sample = {"sat_img": sat_img}
        img_sample["gt_img"] = gt_img

        return img_sample