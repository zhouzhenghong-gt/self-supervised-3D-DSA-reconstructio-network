import os
from time import time
import cv2
import torch
import torch.nn.functional as F
import pytorch_ssim
import math
import re

import numpy as np
import argparse
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import scipy.io as scio
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from load_data import DSAReconDataset

from net.U_Net3D import unet_3D
from utils.project import proj_make_3dinput_v2, resize_image_itk

def threshold_CTA_mask(cta_image, HU_window=np.array([-263.,553.])):
    th_cta_image = (cta_image - HU_window[0])/(HU_window[1] - HU_window[0])
    th_cta_image[th_cta_image < 0] = 0
    th_cta_image[th_cta_image > 1] = 1
    th_cta_image_mask = th_cta_image
    # th_cta_image_mask = (th_cta_image*255).astype('uint8')
    # th_cta_image_mask = (th_cta_image*1023).astype('uint16')
    return th_cta_image_mask

if __name__ == '__main__':
    # validate parameter
    parser = argparse.ArgumentParser(description='DSA 3D Reconstruction validate')
    parser.add_argument('--val_input_path', type=str,
                        help='2d input image and 3d label of training set/validation set')
    parser.add_argument('--module_path', type=str,
                        help='trained module path')
    parser.add_argument('--predict0_path', type=str,
                        help='path of saving predicted niigzfile ')
    parser.add_argument('--stage', type=int, default=1,
                        help='the number of stage of reconstruction network')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--last_stage_path', type=str, default=None,
                        help='the path where the output of the previous/last stage of the network is saved')
    parser.add_argument("--view_num", type=int, default=8,
                        help='number of views for the image input')
    parser.add_argument("--save_niigz", action='store_true')  

    args = parser.parse_args()


    # parameter
    num_workers = args.num_workers
    pin_memory = False
    stage = args.stage
    views = args.view_num
    perangle = 180/views
    input_path = args.val_input_path 
    input_path_2d = os.path.join(input_path,'traindata')
    module_path = args.module_path 
    predict0_path = args.predict0_path 

    # get label_dir
    label_dir = os.listdir(input_path)
    for ii, i in enumerate(label_dir):
        if not i.startswith('traingt'):
            label_dir.pop(ii)
    for ii, i in enumerate(label_dir):
        label_dir[ii] = os.path.join(input_path,i)
    perangle = 180/views

    all_mse = np.zeros((len(label_dir)))
    all_mae = np.zeros((len(label_dir)))
    all_ssim = np.zeros((len(label_dir)))
    all_ssim1 = np.zeros((len(label_dir)))
    all_psnr = np.zeros((len(label_dir)))

    # define network
    net = unet_3D(in_channels=views).cuda()
    net.load_state_dict(torch.load(module_path))
    net.eval()

    # validate
    for file_cnt, file in enumerate(label_dir):

        file_index = int(re.findall('(\d+)',file)[-1])
        start_time = time()

        size = [128, 256, 256]
        crop_size = [32, 256, 256]
        # load 3D images
        image = sitk.ReadImage(file)
        image = resize_image_itk(image, (size[2], size[1], size[0]),resamplemethod=sitk.sitkLinear)
        image_array = sitk.GetArrayFromImage(image)
        # windows
        image_array = threshold_CTA_mask(image_array, HU_window=np.array([1500.,10000.]))

        # get crop chunk list
        index_list = []
        stride = [16, 120, 120]
        I = math.ceil((size[0]-crop_size[0])/stride[0])
        J = math.ceil((size[1]-crop_size[1])/stride[1])
        K = math.ceil((size[2]-crop_size[2])/stride[2])
        for i in range(I+1):
            for j in range(J+1):
                for k in range(K+1):
                    if (i*stride[0]+crop_size[0])>size[0]:
                        index_i1 = size[0]-crop_size[0]
                    else:
                        index_i1 = i*stride[0]

                    if (j*stride[1]+crop_size[1])>size[1]:
                        index_j1 = size[1]-crop_size[1]
                    else:
                        index_j1 = j*stride[1]

                    if (k*stride[2]+crop_size[2])>size[2]:
                        index_k1 = size[2]-crop_size[2]
                    else:
                        index_k1 = k*stride[2]
                    index_list.append([index_i1, index_j1,index_k1])
                    
        ## get input list
        projs_array_list = []
        projs = np.zeros((views, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
        for ii in index_list:
            start_slice0 = ii[0]
            start_slice1 = ii[1]
            start_slice2 = ii[2]
            end_slice0 = start_slice0 + crop_size[0]
            end_slice1 = start_slice1 + crop_size[1]
            end_slice2 = start_slice2 + crop_size[2]
            start_slice = [start_slice0/size[0], start_slice1/size[1], start_slice2/size[2]]
            crop_slice = [crop_size[0] / size[0], crop_size[1] / size[1], crop_size[2] / size[2]]
            # load 2D projections
            projs = np.zeros((views, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
            ## load 2D projections
            for ii in range(views):
                proj_temp = cv2.imread(input_path_2d + '/'+str(views)+'view_low/train'+str(file_index)+'_'+str(ii)+'.jpg',0)
                proj_temp = proj_temp - np.min(proj_temp)
                proj_temp = proj_temp / np.max(proj_temp)
                projs[ii,:,:,:] = proj_make_3dinput_v2(proj_temp, perangle*ii+perangle, start_slice, crop_slice)
            projs_array_list.append(projs)
            projs = np.zeros((views, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)

        ## get pred chunk list
        outputs_list = []
        with torch.no_grad():
            for projs_array in projs_array_list:

                projs_tensor = torch.from_numpy(projs_array).cuda()
                projs_tensor = projs_tensor.unsqueeze(dim=0) # batch

                outputs = net(projs_tensor)
                outputs = outputs.squeeze()

                outputs_list.append(outputs.cpu().detach().numpy())
                del outputs

        ## concat and get full prediction
        pred = np.zeros((size[0], size[1], size[2]), dtype=np.float32)
        pred_cnt = np.zeros((size[0], size[1], size[2]), dtype=np.float32)
        for num, ii in enumerate(index_list):
            start_slice0 = ii[0]
            start_slice1 = ii[1]
            start_slice2 = ii[2]
            end_slice0 = start_slice0 + crop_size[0]
            end_slice1 = start_slice1 + crop_size[1]
            end_slice2 = start_slice2 + crop_size[2]
            pred_cnt[start_slice0:end_slice0, start_slice1:end_slice1, start_slice2:end_slice2] = pred_cnt[start_slice0:end_slice0, start_slice1:end_slice1, start_slice2:end_slice2] + 1
            pred[start_slice0:end_slice0, start_slice1:end_slice1, start_slice2:end_slice2] = pred[start_slice0:end_slice0, start_slice1:end_slice1, start_slice2:end_slice2] + outputs_list[num]
        pred = pred/pred_cnt

        # save prediction as nii.gz file
        if args.save_niigz:
            pred_nii = sitk.GetImageFromArray(pred)
            pred_nii = resize_image_itk(pred_nii, (512, 512, 395),resamplemethod=sitk.sitkLinear)
            sitk.WriteImage(pred_nii, predict0_path + '/predict'+str(file_index)+'.nii.gz')  

        # psnr
        psnr_pred = psnr(image_array, pred)
        # SSIM
        ssim_pred = ssim(image_array, pred)
        print(str(file_index))
        print('mse:', np.mean((pred - image_array) ** 2, dtype=np.float64))
        print('mae:', np.mean(abs(pred - image_array), dtype=np.float64))
        print('ssim:', ssim_pred)
        print('psnr:', psnr_pred)
        all_mse[file_cnt] = np.mean((pred - image_array) ** 2, dtype=np.float64)
        all_mae[file_cnt] = np.mean(abs(pred - image_array), dtype=np.float64)
        all_ssim[file_cnt] = ssim_pred
        all_psnr[file_cnt] = psnr_pred

        pred_ = Variable(torch.from_numpy(pred)).cuda()
        image_array_ = Variable(torch.from_numpy(image_array)).cuda()
        pred_ = pred_.unsqueeze(0)
        pred_ = pred_.unsqueeze(0)
        image_array_  = image_array_.unsqueeze(0)
        image_array_  = image_array_.unsqueeze(0)
        print('ssim1:', pytorch_ssim.ssim3D(image_array_, pred_).item())
        all_ssim1[file_cnt] = pytorch_ssim.ssim3D(image_array_, pred_).item()

        speed = time() - start_time

        with open(predict0_path + '/result_UNet.txt','a') as f:
            f.write(str(file_index)+'  mse:'+str(all_mse[file_cnt])+'  mae:'+str(all_mae[file_cnt])+ '  ssim:'+str(all_ssim[file_cnt])+'  ssim1:'+str(all_ssim1[file_cnt])+'  psnr:'+str(all_psnr[file_cnt])+'\n')

        print('this case use {:.3f} s'.format(speed))
        print('-----------------------')
    print('mean_mse:', np.mean(all_mse))
    print('mean_mae:', np.mean(all_mae))
    print('mean_psnr:', np.mean(all_psnr))
    print('mean_ssim:', np.mean(all_ssim))
    print('mean_ssim1:', np.mean(all_ssim1))
    with open(predict0_path + '/result_UNet.txt','a') as f:
        f.write('mean:'+'  mse:'+str(np.mean(all_mse))+'  mae:'+str(np.mean(all_mae))+ '  ssim:'+str(np.mean(all_ssim))+'  ssim1:'+str(np.mean(all_ssim1))+'  psnr:'+str(np.mean(all_psnr))+'\n')
