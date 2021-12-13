import os
import numpy as np
import SimpleITK as sitk
import cv2
import torch
import random
from torch.utils.data import Dataset

from utils.project import proj_make_3dinput_v2

def threshold_CTA_mask(cta_image, HU_window=np.array([-263.,553.])):
    th_cta_image = (cta_image - HU_window[0])/(HU_window[1] - HU_window[0])
    th_cta_image[th_cta_image < 0] = 0
    th_cta_image[th_cta_image > 1] = 1
    th_cta_image_mask = th_cta_image
    # th_cta_image_mask = (th_cta_image*255).astype('uint8')
    # th_cta_image_mask = (th_cta_image*1023).astype('uint16')
    return th_cta_image_mask

class DSAReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""
    def __init__(self, stage, num_views, input_path, output_path, last_path = None):
        """
        Args:
            file_list (string): Path to the csv file with annotations.
            data_root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.stage = stage
        self.input_path = input_path
        self.output_path = output_path
        self.last_path = last_path

        self.num_views = num_views

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, index):

        if self.stage == 1:
            size = [128, 256, 256]
            crop_size = [32, 256, 256]
        elif self.stage == 2:
            size = [395, 512, 512]
            crop_size = [32, 512, 512]

        views = self.num_views
        proj_path = os.path.join(self.input_path, os.listdir(self.input_path)[index])

        # get ramdom crop
        start_slice0 = random.randint(0, size[0] - crop_size[0])
        end_slice0 = start_slice0 + crop_size[0]
        start_slice1 = random.randint(0, size[1] - crop_size[1])
        end_slice1 = start_slice1 + crop_size[1]
        start_slice2 = random.randint(0, size[2] - crop_size[2])
        end_slice2 = start_slice2 + crop_size[2]
        start_slice = [start_slice0/size[0], start_slice1/size[1], start_slice2/size[2]]
        crop_slice = [crop_size[0] / size[0], crop_size[1] / size[1], crop_size[2] / size[2]]

        # load 2D projections and unproject to 3D input
        perangle = 180/views
        if self.stage == 1:
            projs = np.zeros((views, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
        elif self.stage > 1:
            projs = np.zeros((views+1, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
        image_array_proj = np.zeros((views, crop_size[0], crop_size[1]), dtype=np.float32)
        for ii in range(views):
            if self.stage == 1:
                proj_temp = cv2.imread(self.output_path + '/traindata/'+str(views)+'view_low/train'+str(int(proj_path[-2:]))+'_'+str(ii)+'.jpg',0)
            elif self.stage > 1:
                proj_temp = cv2.imread(self.output_path + '/traindata/'+str(views)+'view/train'+str(int(proj_path[-2:]))+'_'+str(ii)+'.jpg',0)
            proj_temp = proj_temp - np.min(proj_temp)
            proj_temp = proj_temp / np.max(proj_temp)
            projs[ii,:,:,:] = proj_make_3dinput_v2(proj_temp, perangle*ii+perangle, start_slice, crop_slice)
            image_array_proj[ii,:,:] = proj_temp[start_slice0:end_slice0,:]
        
        # use last stage output as input
        if self.stage > 1:
            assert self.last_path==None
            image_nii = sitk.ReadImage(self.last_path + '/predict'+str(int(proj_path[-2:]))+'.nii.gz')
            projs[views] = sitk.GetArrayFromImage(image_nii)[start_slice0:end_slice0, start_slice1:end_slice1, start_slice2:end_slice2] 

        image_array_proj = torch.from_numpy(image_array_proj).float()
        projs = torch.from_numpy(projs).float()
        return (projs, image_array_proj)

