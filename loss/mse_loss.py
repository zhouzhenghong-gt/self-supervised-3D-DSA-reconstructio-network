import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import SimpleITK as sitk

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸 h,w,d
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)# h,w,d
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

def oblique_project1(pred, angle = 67.5):
    # 斜投影会损失一部分边界信息！
    # 相当于确定max(1)是0度！！
    angle1 = angle
    h = pred.shape[0+1]
    w = pred.shape[1+1]
    l = pred.shape[2+1]
    if angle <= 45:
        label = pred
        l1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred = F.interpolate(pred.unsqueeze(0), size=[h,w,l1], mode='trilinear', align_corners=True)
        pred = pred.squeeze(0)

        pred_proj = torch.zeros((pred.size()[0], h, l1+w-1))
        for i in range(l1+w-1):
            pred_proj[:,:,i] = (torch.diagonal(pred, i-w+1, 1+1, 2+1)).max(1+1)[0]

        L = round((w**2+l1**2)**0.5*angle/45) #???
        pred_proj = F.interpolate(pred_proj.unsqueeze(0), size=[h,L], mode='bilinear', align_corners=True)
        pred_proj = pred_proj.squeeze(0)
        pred_proj = pred_proj[:,:,round(L/2)-int(l/2):round(L/2)+int(l/2)] # 损失一部分信息？？
    elif (angle > 45) & (angle < 90):
        label = pred
        angle = 90-angle
        w1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred = F.interpolate(pred.unsqueeze(0), size=[h,w1,l], mode='trilinear', align_corners=True)
        pred = pred.squeeze(0)

        pred_proj = torch.zeros((pred.size()[0], h, l+w1-1))
        for i in range(l+w1-1):
            pred_proj[:,:,i] = (torch.diagonal(pred, i-w1+1, 1+1, 2+1)).max(1+1)[0]

        L = round((w1**2+l**2)**0.5*angle/45) #???
        pred_proj = F.interpolate(pred_proj.unsqueeze(0), size=[h,L], mode='bilinear', align_corners=True)
        pred_proj = pred_proj.squeeze(0)
        pred_proj = pred_proj[:,:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
    elif angle == 90:
        label = pred
        # pred = np.flip(pred, 1)
        pred = torch.flip(pred, [1+1])
        pred_proj = pred.max(2+1)[0]
    elif (angle > 90) & (angle <= 135):
        label = pred
        angle = angle - 90
        pred = torch.flip(pred, [1+1])
        w1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred = F.interpolate(pred.unsqueeze(0), size=[h,w1,l], mode='trilinear', align_corners=True)
        pred = pred.squeeze(0)

        pred_proj = torch.zeros((pred.size()[0], h, l+w1-1))
        for i in range(l+w1-1):
            pred_proj[:,:,i] = (torch.diagonal(pred, i-w1+1, 1+1, 2+1)).max(1+1)[0]

        L = round((w1**2+l**2)**0.5*angle/45) #???
        pred_proj = F.interpolate(pred_proj.unsqueeze(0), size=[h,L], mode='bilinear', align_corners=True)
        pred_proj = pred_proj.squeeze(0)
        pred_proj = pred_proj[:,:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
        pred_proj = torch.flip(pred_proj, [1+1])
    elif (angle > 135) & (angle < 180):
        label = pred
        angle = 180 - angle
        pred = torch.flip(pred, [1+1])
        l1 = round((1.0/math.tan(math.radians(angle)))*l)
        pred = F.interpolate(pred.unsqueeze(0), size=[h,w,l1], mode='trilinear', align_corners=True)
        pred = pred.squeeze(0)

        pred_proj = torch.zeros((pred.size()[0], h, l1+w-1))
        for i in range(l1+w-1):
            pred_proj[:,:,i] = (torch.diagonal(pred, i-w+1, 1+1, 2+1)).max(1+1)[0]

        L = round((w**2+l1**2)**0.5*angle/45) #???
        pred_proj = F.interpolate(pred_proj.unsqueeze(0), size=[h,L], mode='bilinear', align_corners=True)
        pred_proj = pred_proj.squeeze(0)
        pred_proj = pred_proj[:,:,round(L/2)-int(l/2):round(L/2)+int(l/2)]
        pred_proj = torch.flip(pred_proj, [1+1])
    elif angle == 180:
        label = pred
        pred = torch.flip(pred, [2+1])
        pred_proj = pred.max(1+1)[0]

    # # test
    # pred0 = pred_proj[0,:,:].cpu()
    # pred0 = pred0.detach().numpy()  # 一块的label可能会与输入的max不同，因为输入是先裁剪的
    # label0 = label[0,:,:,:].cpu()
    # label0 = label0.detach().numpy()

    # transforms = Rotate(angle=math.radians(angle1))
    # label_r = apply_transform(transforms,label0.astype(np.float))
    # concat = np.concatenate((pred0,label_r.max(1)), axis=1)   #相当于确定max(1)是0度！！
    # plt.imshow(concat, cmap="gray")
    # plt.savefig('/home/zhenghongzhou/repo/DSA_reconstruct/train8_U-net/result_17_weak8view/pred_proj_'+str(angle1)+'.jpg')

    return pred_proj

class MSELoss(nn.Module):
    def __init__(self, views):
        super().__init__()

        self.loss = nn.MSELoss()
        self.views = views

    def forward(self, pred, target):
        h = pred.shape[0+1]
        w = pred.shape[1+1]
        l = pred.shape[2+1]
        perangle = 180/self.views
        pred_proj = torch.zeros((pred.size()[0], self.views, h, l)).cuda()
        for i in range(self.views):
            pred_proj[:,i,:,:] = oblique_project1(pred, perangle*i+perangle)
        loss = self.loss(pred_proj, target)
        return loss, pred_proj


