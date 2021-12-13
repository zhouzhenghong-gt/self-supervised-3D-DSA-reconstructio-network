import cv2
import math
import SimpleITK as sitk
import numpy as np

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def oblique_project(pred, angle = 67.5):
    """
    This function projects 3d voxel data into 2d data at different angles/views

    :param pred: 3d voxel input 
    :param angle: the angle of different view. set max(1) as 0.
    :return pred_proj: 2d output 
    """
    cv2.setNumThreads(0)
    angle1 = angle
    h = pred.shape[0]
    w = pred.shape[1]
    l = pred.shape[2]
    if angle <= 45:
        label = pred
        l1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred_itk = sitk.GetImageFromArray(pred)
        pred_itk = resize_image_itk(pred_itk, (l1, w, h),resamplemethod=sitk.sitkLinear)
        pred = sitk.GetArrayFromImage(pred_itk)

        pred_proj = np.zeros((h, l1+w-1), dtype=np.float32)
        for i in range(l1+w-1):
            pred_proj[:,i] = pred.diagonal(i-w+1,1,2).max(1)

        L = round((w**2+l1**2)**0.5*angle/45)
        pred_proj = cv2.resize(pred_proj,(L, h))
        pred_proj = pred_proj[:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
    elif (angle > 45) & (angle < 90):
        label = pred
        angle = 90-angle
        w1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred_itk = sitk.GetImageFromArray(pred)
        pred_itk = resize_image_itk(pred_itk, (l, w1, h),resamplemethod=sitk.sitkLinear)
        pred = sitk.GetArrayFromImage(pred_itk)

        pred_proj = np.zeros((h, l+w1-1), dtype=np.float32)
        for i in range(l+w1-1):
            pred_proj[:,i] = pred.diagonal(i-w1+1,1,2).max(1)

        L = round((w1**2+l**2)**0.5*angle/45)
        pred_proj = cv2.resize(pred_proj,(L, h))
        pred_proj = pred_proj[:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
    elif angle == 90:
        label = pred
        pred = np.flip(pred, 1)
        pred_proj = pred.max(2)
    elif (angle > 90) & (angle <= 135):
        label = pred
        angle = angle - 90
        pred = np.flip(pred, 1)
        w1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred_itk = sitk.GetImageFromArray(pred)
        pred_itk = resize_image_itk(pred_itk, (l, w1, h),resamplemethod=sitk.sitkLinear)
        pred = sitk.GetArrayFromImage(pred_itk)

        pred_proj = np.zeros((h, l+w1-1), dtype=np.float32)
        for i in range(l+w1-1):
            pred_proj[:,i] = pred.diagonal(i-w1+1,1,2).max(1)

        L = round((w1**2+l**2)**0.5*angle/45)
        pred_proj = cv2.resize(pred_proj,(L, h))
        pred_proj = pred_proj[:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
        pred_proj = np.flip(pred_proj, 1)
    elif (angle > 135) & (angle < 180):
        label = pred
        angle = 180 - angle
        pred = np.flip(pred, 1)
        l1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred_itk = sitk.GetImageFromArray(pred)
        pred_itk = resize_image_itk(pred_itk, (l1, w, h),resamplemethod=sitk.sitkLinear)
        pred = sitk.GetArrayFromImage(pred_itk)

        pred_proj = np.zeros((h, l1+w-1), dtype=np.float32)
        for i in range(l1+w-1):
            pred_proj[:,i] = pred.diagonal(i-w+1,1,2).max(1)

        L = round((w**2+l1**2)**0.5*angle/45)
        pred_proj = cv2.resize(pred_proj,(L, h))
        pred_proj = pred_proj[:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
        pred_proj = np.flip(pred_proj, 1)
    elif angle == 180:
        label = pred
        pred = np.flip(pred, 2)
        pred_proj = pred.max(1)

    return pred_proj

def proj_make_3dinput_v2(project, angle = 15, start_slice = [0,0,0], crop_slice = [0.75, 0.625, 0.625]):
    """
    This function unprojects 2d data into 3d voxel at different angles/views and do crop.

    :param project: 2d image input 
    :param angle: the angle of different view. set max(1) as 0.
    :param start_slice: start slice of three dimension
    :param crop_slice: crop ratio of three dimension
    :return pred_proj: 3d output 
    """
    angle1 = angle
    h = project.shape[0]
    w = project.shape[1]
    l = project.shape[1]
    if angle <= 45:
        label = project
        l1 = round((1.0/math.tan(math.radians(angle)))*l)

        L = round((w**2+l1**2)**0.5*angle/45)
        p = round((L-l)/2)
        project = np.pad(project,((0,0),(p,p)),'constant', constant_values=(0,0))
        pred_proj = cv2.resize(project,(l1+w-1, h))
        # crop
        s1 = round(start_slice[1]*w)
        s2 = round(start_slice[2]*l1)
        pred_proj = pred_proj[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),:]
        input3d = np.zeros((round(crop_slice[0]*h), round(crop_slice[1]*w), round(crop_slice[2]*l1)), dtype=np.float32)
        for i in range(round(crop_slice[1]*w+crop_slice[2]*l1-1)):
            relen = input3d.diagonal(round(i-crop_slice[1]*w+1),1,2).shape[1]
            row, col = np.diag_indices(relen)
            if i < (input3d.shape[1]-1):
                input3d[:,row-(i-input3d.shape[1]+1), col] = np.expand_dims(pred_proj[:,i+w-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
            elif i >= (input3d.shape[1]-1):
                input3d[:,row, col+(i-input3d.shape[1]+1)] = np.expand_dims(pred_proj[:,i+w-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)

        input3d_itk = sitk.GetImageFromArray(input3d)
        input3d_itk = resize_image_itk(input3d_itk, (round(crop_slice[2]*l), round(crop_slice[1]*w), round(crop_slice[0]*h)),resamplemethod=sitk.sitkLinear)
        input3d = sitk.GetArrayFromImage(input3d_itk)

    elif (angle > 45) & (angle < 90):
        label = project
        angle = 90-angle
        w1 = round((1.0/math.tan(math.radians(angle)))*l)
        
        L = round((w1**2+l**2)**0.5*angle/45)
        p = round((L-l)/2)
        project = np.pad(project,((0,0),(p,p)),'constant', constant_values=(0,0))
        pred_proj = cv2.resize(project,(l+w1-1, h))
        # crop
        s1 = round(start_slice[1]*w1)
        s2 = round(start_slice[2]*l)
        pred_proj = pred_proj[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),:]
        input3d = np.zeros((round(crop_slice[0]*h), round(crop_slice[1]*w1), round(crop_slice[2]*l)), dtype=np.float32)
        for i in range(round(crop_slice[1]*w1+crop_slice[2]*l-1)):
            relen = input3d.diagonal(round(i-crop_slice[1]*w1+1),1,2).shape[1]
            row, col = np.diag_indices(relen)
            if i < (input3d.shape[1]-1):
                input3d[:,row-(i-input3d.shape[1]+1), col] = np.expand_dims(pred_proj[:,i+w1-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
            elif i >= (input3d.shape[1]-1):
                input3d[:,row, col+(i-input3d.shape[1]+1)] = np.expand_dims(pred_proj[:,i+w1-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)

        input3d_itk = sitk.GetImageFromArray(input3d)
        input3d_itk = resize_image_itk(input3d_itk, (round(crop_slice[2]*l), round(crop_slice[1]*w), round(crop_slice[0]*h)),resamplemethod=sitk.sitkLinear)
        input3d = sitk.GetArrayFromImage(input3d_itk)
    elif angle == 90:
        label = project
        project = np.flip(project, 1)
        pred_proj = project[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),
                            round(start_slice[1]*w):round((start_slice[1]+crop_slice[1])*w)]
        input3d = np.expand_dims(pred_proj, 2).repeat(pred_proj.shape[1], axis=2)
    elif (angle > 90) & (angle <= 135):
        label = project
        angle = angle - 90
        project = np.flip(project, 1)
        w1 = round((1.0/math.tan(math.radians(angle)))*l)

        L = round((w1**2+l**2)**0.5*angle/45)
        p = round((L-l)/2)
        project = np.pad(project,((0,0),(p,p)),'constant', constant_values=(0,0))
        pred_proj = cv2.resize(project,(l+w1-1, h))
        # crop
        start_slice[1] = 1 - (start_slice[1] + crop_slice[1])
        s1 = round(start_slice[1]*w1)
        s2 = round(start_slice[2]*l)
        pred_proj = pred_proj[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),:]
        input3d = np.zeros((round(crop_slice[0]*h), round(crop_slice[1]*w1), round(crop_slice[2]*l)), dtype=np.float32)
        for i in range(round(crop_slice[1]*w1+crop_slice[2]*l-1)):
            relen = input3d.diagonal(round(i-crop_slice[1]*w1+1),1,2).shape[1]
            row, col = np.diag_indices(relen)
            if i < (input3d.shape[1]-1):
                input3d[:,row-(i-input3d.shape[1]+1), col] = np.expand_dims(pred_proj[:,i+w1-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
            elif i >= (input3d.shape[1]-1):
                input3d[:,row, col+(i-input3d.shape[1]+1)] = np.expand_dims(pred_proj[:,i+w1-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
        input3d = np.flip(input3d, 1)
        input3d_itk = sitk.GetImageFromArray(input3d)
        input3d_itk = resize_image_itk(input3d_itk, (round(crop_slice[2]*l), round(crop_slice[1]*w), round(crop_slice[0]*h)),resamplemethod=sitk.sitkLinear)
        input3d = sitk.GetArrayFromImage(input3d_itk)
        start_slice[1] = 1 - (start_slice[1] + crop_slice[1]) 
    elif (angle > 135) & (angle < 180):
        label = project
        angle = 180 - angle
        project = np.flip(project, 1)
        l1 = round((1.0/math.tan(math.radians(angle)))*l)

        L = round((w**2+l1**2)**0.5*angle/45)
        p = round((L-l)/2)
        project = np.pad(project,((0,0),(p,p)),'constant', constant_values=(0,0))
        pred_proj = cv2.resize(project,(l1+w-1, h))
        # crop
        start_slice[1] = 1 - (start_slice[1] + crop_slice[1])
        s1 = round(start_slice[1]*w)
        s2 = round(start_slice[2]*l1)
        pred_proj = pred_proj[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),:]
        input3d = np.zeros((round(crop_slice[0]*h), round(crop_slice[1]*w), round(crop_slice[2]*l1)), dtype=np.float32)
        for i in range(round(crop_slice[1]*w+crop_slice[2]*l1-1)):
            relen = input3d.diagonal(round(i-crop_slice[1]*w+1),1,2).shape[1]
            row, col = np.diag_indices(relen)
            if i < (input3d.shape[1]-1):
                input3d[:,row-(i-input3d.shape[1]+1), col] = np.expand_dims(pred_proj[:,i+w-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
            elif i >= (input3d.shape[1]-1):
                input3d[:,row, col+(i-input3d.shape[1]+1)] = np.expand_dims(pred_proj[:,i+w-s1+s2-input3d.shape[1]],1).repeat(relen, axis=1)
        input3d = np.flip(input3d, 1)
        input3d_itk = sitk.GetImageFromArray(input3d)
        input3d_itk = resize_image_itk(input3d_itk, (round(crop_slice[2]*l), round(crop_slice[1]*w), round(crop_slice[0]*h)),resamplemethod=sitk.sitkLinear)
        input3d = sitk.GetArrayFromImage(input3d_itk)
        start_slice[1] = 1 - (start_slice[1] + crop_slice[1])
    elif angle == 180:
        label = project
        project = np.flip(project, 1)
        pred_proj = project[round(start_slice[0]*h):round((start_slice[0]+crop_slice[0])*h),
                            round(start_slice[2]*l):round((start_slice[2]+crop_slice[2])*l)]
        input3d = np.expand_dims(pred_proj, 1).repeat(pred_proj.shape[1], axis=1)
    return input3d
