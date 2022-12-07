import torch
import torch.nn as nn

class DoubleConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x) 
    
class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
init_channels = 64
Input_Image_Channels = 1
Num_Classes = 4


class Model_4(nn.Module):         #### SA and LA
    def __init__(self):
        super(Model_4, self).__init__()
        
        self.conv1_la = DoubleConv_2d(Input_Image_Channels, init_channels)
        self.conv2_la = DoubleConv_2d(init_channels, 2*init_channels)
        self.conv3_la = DoubleConv_2d(2*init_channels, 4*init_channels)
        self.conv4_la = DoubleConv_2d(4*init_channels, 8*init_channels)
        self.conv5_la = DoubleConv_2d(8*init_channels, 16*init_channels)
        self.conv6_la = DoubleConv_2d(16*init_channels, 16*init_channels)
        
        self.conv_u1_la = DoubleConv_2d(16*init_channels, 8*init_channels)
        self.conv_u2_la = DoubleConv_2d(8*init_channels + 16*init_channels, 4*init_channels)
        self.conv_u3_la = DoubleConv_2d(4*init_channels + 8*init_channels, 2*init_channels)
        self.conv_u4_la = DoubleConv_2d(2*init_channels + 4*init_channels ,init_channels)
        self.conv_u5_la = DoubleConv_2d(init_channels   + 2*init_channels,init_channels)
        self.conv_u6_la = DoubleConv_2d(init_channels   + init_channels,init_channels)
        
        self.out_la = OutConv_2d(init_channels, Num_Classes)
          
        self.max_pool_2d = nn.MaxPool2d(2)
        self.activation = torch.nn.Softmax(dim = 1)
        self.up_sample_2d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.Drop_Out= nn.Dropout(p=0.40)
        
    def forward(self,x_la):
        
           ############### STAGE -------->> 1 #################
           #####################################################
        ### Encoder Design for LA ###
        conv1_LA = self.conv1_la(x_la)   
        conv1_LA_pool= self.max_pool_2d(conv1_LA)
           ############### STAGE -------->> 2 #################
           #####################################################
           
        # ### Encoder Design for LA ###
        conv2_LA = self.conv2_la(conv1_LA_pool)
        conv2_LA_pool = self.max_pool_2d(conv2_LA)
       
           ############### STAGE -------->> 3 #################
           #####################################################
           
        conv3_LA = self.conv3_la(conv2_LA_pool)
        conv3_LA_pool= self.max_pool_2d(conv3_LA)

      
        
           ############### STAGE -------->> 4 #################
           #####################################################
           
        conv4_LA = self.conv4_la(conv3_LA_pool)
        conv4_LA_pool = self.max_pool_2d(conv4_LA)
        
      
           ############### STAGE -------->> 5 #################
           #####################################################
           
        conv5_LA = self.conv5_la(conv4_LA_pool)
        conv5_LA_pool = self.max_pool_2d(conv5_LA)
        
       
           ############### STAGE -------->> 5  BOTTLENECK #################
           #####################################################
        
        conv6_LA = self.conv6_la(conv5_LA_pool)  ### 8 x 8 x 1024
      
       
        conv6_LA = self.Drop_Out(conv6_LA)

        ## Decoder Design for LA ####
        
        u1_LA = self.up_sample_2d(conv6_LA)
        u1_LA = self.conv_u1_la(u1_LA)
        u1_LA = torch.cat([conv5_LA, u1_LA], dim=1) ### 
                
        
        u2_LA = self.up_sample_2d(u1_LA)
        u2_LA = self.conv_u2_la(u2_LA)
        u2_LA = torch.cat([conv4_LA, u2_LA], dim=1) ### 
        
        u3_LA = self.up_sample_2d(u2_LA)
        u3_LA = self.conv_u3_la(u3_LA)
        u3_LA = torch.cat([conv3_LA, u3_LA], dim=1) ### 
        
        u4_LA = self.up_sample_2d(u3_LA)
        u4_LA = self.conv_u4_la(u4_LA)
        u4_LA = torch.cat([conv2_LA, u4_LA], dim=1) ### 
        
        u5_LA = self.up_sample_2d(u4_LA)
        u5_LA = self.conv_u5_la(u5_LA)
        u5_LA = torch.cat([conv1_LA, u5_LA], dim=1) ### 

        u6_LA = self.conv_u6_la(u5_LA)
        
        out_LA = self.out_la(u6_LA)
        
        #out_LA = self.activation(out_LA)
        return out_LA
    
# def model() -> Model_4:
#     model = Model_4()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256),(Input_Image_Channels,5,256,256)])

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from typing import List, Union, Tuple


import numpy as np
from scipy import ndimage

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_=256

def LA_to_SA(SA_img,LA_img):
    # Get sizes
    SA_size = (SA_img.GetSize())   ## --> [H,W,C]
    LA_size = (LA_img.GetSize())
    
    # Create a new short axis image the same size as the SA stack.
    new_SA_img = sitk.Image(SA_size, sitk.sitkFloat64)
    
    # Loop over every pixel in the LA image, and put into into the new SA image
    for x in range(0, LA_size[0]):
        for y in range(0, LA_size[1]):
            # Determine the physical location of the LA pixel
            point = LA_img.TransformIndexToPhysicalPoint([x, y, 0])
    
            # Find which index this position maps to in the SA image
            index_SA = SA_img.TransformPhysicalPointToIndex(point)
    
            # Check if the pixel is outside the bounds of the SA image
            if index_SA[0] - 1 < 0 or index_SA[0] + 1 >= SA_img.GetSize()[0]:
                continue
            if index_SA[1] - 1 < 0 or index_SA[1] + 1 >= SA_img.GetSize()[1]:
                continue
            if index_SA[2] - 1 < 0 or index_SA[2] + 1 >= SA_img.GetSize()[2]:
                continue
    
            # Assign the LA pixel to the voxel location in the new SA image
            new_SA_img[index_SA[0], index_SA[1], index_SA[2]] = LA_img[x, y, 0]
    
            # Dilate the intensity (optional)
            new_SA_img[index_SA[0] - 1, index_SA[1], index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1] - 1, index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1], index_SA[2] - 1] = LA_img[x, y, 0]
            new_SA_img[index_SA[0] + 1, index_SA[1], index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1] + 1, index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1], index_SA[2] + 1] = LA_img[x, y, 0]
    return new_SA_img


def resample_image_SA(itk_image):
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / original_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    

def resample_image_LA(itk_image):

    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    
def crop_center_3D(img,cropx=256,cropy=256):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
    
def Normalization_2(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
class Dataset_Both_ES(Dataset): 
    def __init__(self, df, images_folder):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        
        
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        img_SA = Normalization_2(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
        
        temp_SA=np.zeros([4,org_dim3,DIM_,DIM_])
        temp_SA[0,:,:,:][np.where(img_SA_gt==1)]=1
        temp_SA[1,:,:,:][np.where(img_SA_gt==2)]=1
        temp_SA[2,:,:,:][np.where(img_SA_gt==3)]=1
        temp_SA[3,:,:,:][np.where(img_SA_gt==0)]=1
   
        #####    LA Images #####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        
        
        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        
        img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA = Normalization_1(img_LA)
        img_LA = Normalization_2(img_LA)
        
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        
        
        temp_LA = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_LA[0,:,:,:][np.where(img_LA_gt==1)]=1
        temp_LA[1,:,:,:][np.where(img_LA_gt==2)]=1
        temp_LA[2,:,:,:][np.where(img_LA_gt==3)]=1
        temp_LA[3,:,:,:][np.where(img_LA_gt==0)]=1
        
        ### sa_to_la mapping ####
        
#        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
#        img_SA_path = img_path_SA +'_SA_ES.nii.gz'
#        img_SA_1 = sitk.ReadImage(img_SA_path)
#        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
#        img_LA_path = img_path_LA +'_LA_ES.nii.gz'
#        img_LA_1 = sitk.ReadImage(img_LA_path)
#        
#        new_SA_img = LA_to_SA(img_SA_1,img_LA_1)
#        new_SA_img = resample_image_SA(new_SA_img)
#        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
#        
#
#        org_dim3 = new_SA_img.shape[0]
#        org_dim1 = new_SA_img.shape[1]
#        org_dim2 = new_SA_img.shape[2] 
#
#        new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
#        
#        new_SA_img = Normalization_1(new_SA_img)
#        new_SA_img = Normalization_2(new_SA_img)
#        new_SA_img = np.expand_dims(new_SA_img, axis=0)
#
#        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,new_SA_img,self.images_name[index]
        
        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,self.images_name[index]
           
def Data_Loader_Both_ES(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_Both_ES(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_Both_ED(Dataset): 
    def __init__(self, df, images_folder):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        
        
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        img_SA = Normalization_2(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
        
        temp_SA=np.zeros([4,org_dim3,DIM_,DIM_])
        temp_SA[0,:,:,:][np.where(img_SA_gt==1)]=1
        temp_SA[1,:,:,:][np.where(img_SA_gt==2)]=1
        temp_SA[2,:,:,:][np.where(img_SA_gt==3)]=1
        temp_SA[3,:,:,:][np.where(img_SA_gt==0)]=1
   
        #####    LA Images #####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        
        
        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        
        img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA = Normalization_1(img_LA)
        img_LA = Normalization_2(img_LA)
        
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        
        
        temp_LA = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_LA[0,:,:,:][np.where(img_LA_gt==1)]=1
        temp_LA[1,:,:,:][np.where(img_LA_gt==2)]=1
        temp_LA[2,:,:,:][np.where(img_LA_gt==3)]=1
        temp_LA[3,:,:,:][np.where(img_LA_gt==0)]=1
        
        ### sa_to_la mapping ####
        
#        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
#        img_SA_path = img_path_SA +'_SA_ED.nii.gz'
#        img_SA_1 = sitk.ReadImage(img_SA_path)
#        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
#        img_LA_path = img_path_LA +'_LA_ED.nii.gz'
#        img_LA_1 = sitk.ReadImage(img_LA_path)
#        
#        new_SA_img = LA_to_SA(img_SA_1,img_LA_1)
#        new_SA_img = resample_image_SA(new_SA_img)
#        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
#        
#
#        org_dim3 = new_SA_img.shape[0]
#        org_dim1 = new_SA_img.shape[1]
#        org_dim2 = new_SA_img.shape[2] 
#
#        new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
#        
#        new_SA_img = Normalization_1(new_SA_img)
#        new_SA_img = Normalization_2(new_SA_img)
#        new_SA_img = np.expand_dims(new_SA_img, axis=0)
#
#        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,new_SA_img,self.images_name[index]
        
        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,self.images_name[index]
           
def Data_Loader_Both_ED(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_Both_ED(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader






   #### Specify all the paths here #####
   
train_imgs='/data/scratch/acw676/MnM/data_2/train/'
val_imgs='/data/scratch/acw676/MnM/data_2/val/'
#test_imgs='/data/scratch/acw676/MnM/test/'

train_csv_path='/data/scratch/acw676/MnM/train.csv'
val_csv_path='/data/scratch/acw676/MnM/val.csv'
#test_csv_path='/data/scratch/acw676/MnM/test.csv'

path_to_save_Learning_Curve='/data/home/acw676/MM/weights/'+'/multi_2D_1'
path_to_save_check_points='/data/home/acw676/MM/weights/'+'/multi_2D_1'
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size=1
Max_Epochs=100
LEARNING_RATE=0.0001
Patience = 4

        #### Import All libraies used for training  #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from Early_Stopping import EarlyStopping
            ### Data_Generators ########
            
            #### The first one will agument and Normalize data and used for training ###
            #### The second will not apply Data augmentaations and only prcocess the data during validation ###


df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

train_loader_ES = Data_Loader_Both_ES(df_train,train_imgs,batch_size)
val_loader_ES = Data_Loader_Both_ES(df_val,val_imgs,batch_size)

train_loader_ED = Data_Loader_Both_ED(df_train,train_imgs,batch_size)
val_loader_ED = Data_Loader_Both_ED(df_val,val_imgs,batch_size)




   ### Load the Data using Data generators and paths specified #####
   #######################################
   
print(len(train_loader_ES)) ### this shoud be = Total_images/ batch size
print(len(val_loader_ES))   ### same here
#print(len(test_loader))   ### same here

print(len(train_loader_ED)) ### this shoud be = Total_images/ batch size
print(len(val_loader_ED))   ### same here
#print(len(test_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses1 = []   # losses of all training epochs
avg_valid_losses1 = []  #losses of all training epochs
avg_valid_DS1 = []  # all training epochs


avg_train_losses2 = []   # losses of all training epochs
avg_valid_losses2 = []  #losses of all training epochs
avg_valid_DS2 = []  # all training epochs

### Next we have all the funcitons which will be called in the main for training ####

Actual_ = 0.5
Not_ = 0.5  

LA_ = 0.5
SA_ = 0.5

RV_ = 0.4
LV_ = 0.3
MYO_ = 0.3 

def generate_label_1(gt):
        gt = torch.argmax(gt,dim =1)    
        temp_ = torch.zeros([gt.shape[0],4,256,256])
        temp_ = temp_.to(device=DEVICE,dtype=torch.float) 
        temp_[:,0,:,:][torch.where(gt==0)]=1
        temp_[:,1,:,:][torch.where(gt==1)]=1
        temp_[:,2,:,:][torch.where(gt==2)]=1
        temp_[:,3,:,:][torch.where(gt==3)]=1
        return temp_
    
### 2- the main training fucntion to update the weights....
def train_fn(loader_train1,loader_train2,loader_valid1, loader_valid2,model, optimizer,loss_fn1, scaler):  ### Loader_1--> ED and Loader2-->ES
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch
    
    train_losses2 = [] # loss of each batch
    valid_losses2 = []  # loss of each batch
    
    
    loop = tqdm(loader_train1)
    model.train()
    for batch_idx, (img_LA,temp_LA,img_SA,temp_SA,label) in enumerate(loop):
        img_LA = img_LA.to(device=DEVICE,dtype=torch.float)  
        temp_LA = temp_LA.to(device=DEVICE,dtype=torch.float)
        img_SA = img_SA.to(device=DEVICE,dtype=torch.float)  
        temp_SA = temp_SA.to(device=DEVICE,dtype=torch.float)
        
        with torch.cuda.amp.autocast():
            out_LA = model(img_LA)    
            loss1 = loss_fn1(out_LA,temp_LA)
            
        # backward
        #loss = (loss1 + loss2)/2
        loss = loss1
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1.append(float(loss))
        
    loop = tqdm(loader_train2)
    #model.train()
    for batch_idx, (img_LA,temp_LA,img_SA,temp_SA,label) in enumerate(loop):
       img_LA = img_LA.to(device=DEVICE,dtype=torch.float)  
       temp_LA = temp_LA.to(device=DEVICE,dtype=torch.float)
       img_SA = img_SA.to(device=DEVICE,dtype=torch.float)  
       temp_SA = temp_SA.to(device=DEVICE,dtype=torch.float)

       with torch.cuda.amp.autocast():
           out_LA = model(img_LA)   
           loss1 = loss_fn1(out_LA,temp_LA) 
           
          
       loss = loss1
       optimizer.zero_grad()
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
        # update tqdm loop
        
       loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
       train_losses2.append(float(loss))

    loop_v = tqdm(loader_valid1)
    model.eval()
    for batch_idx, (img_LA,temp_LA,img_SA,temp_SA,label) in enumerate(loop_v):
        img_LA = img_LA.to(device=DEVICE,dtype=torch.float)  
        temp_LA = temp_LA.to(device=DEVICE,dtype=torch.float)
        img_SA = img_SA.to(device=DEVICE,dtype=torch.float)  
        temp_SA = temp_SA.to(device=DEVICE,dtype=torch.float)

        # forward
        with torch.no_grad(): 
            out_LA = model(img_LA)  
            loss1 = loss_fn1(out_LA,temp_LA)
        # backward
        loss = loss1 
        loop_v.set_postfix(loss = loss.item())
        valid_losses1.append(float(loss))

    loop_v = tqdm(loader_valid2)
    #model.eval()
    for batch_idx, (img_LA,temp_LA,img_SA,temp_SA,label) in enumerate(loop_v):
        img_LA = img_LA.to(device=DEVICE,dtype=torch.float)  
        temp_LA = temp_LA.to(device=DEVICE,dtype=torch.float)
        img_SA = img_SA.to(device=DEVICE,dtype=torch.float)  
        temp_SA = temp_SA.to(device=DEVICE,dtype=torch.float)

        # forward
        with torch.no_grad():
            out_LA = model(img_LA)  
            loss1 = loss_fn1(out_LA,temp_LA)
           
        # backward
        loss = loss1 
        loop_v.set_postfix(loss = loss.item())
        valid_losses2.append(float(loss))
        
    train_loss_per_epoch1 = np.average(train_losses1)
    valid_loss_per_epoch1 = np.average(valid_losses1)
    ## all epochs
    avg_train_losses1.append(train_loss_per_epoch1)
    avg_valid_losses1.append(valid_loss_per_epoch1)
    
    train_loss_per_epoch2 = np.average(train_losses2)
    valid_loss_per_epoch2 = np.average(valid_losses2)
    ## all epochs
    avg_train_losses2.append(train_loss_per_epoch2)
    avg_valid_losses2.append(valid_loss_per_epoch2)
    
    return (train_loss_per_epoch1+train_loss_per_epoch2)/2,(valid_loss_per_epoch1+valid_loss_per_epoch2)/2


    ### 4 - It will check the Dice-Score on each epoch for validation data 
def check_Dice_Score(loader, model, device=DEVICE):
    Dice_score_LA_RV = 0
    Dice_score_LA_MYO = 0
    Dice_score_LA_LV = 0
    Dice_score_BG = 0

    loop = tqdm(loader)
    model.eval()
    for batch_idx, (img_LA,temp_LA,img_SA,temp_SA,label) in enumerate(loop):
        img_LA = img_LA.to(device=DEVICE,dtype=torch.float)  
        temp_LA = temp_LA.to(device=DEVICE,dtype=torch.float)
        img_SA = img_SA.to(device=DEVICE,dtype=torch.float)  
        temp_SA = temp_SA.to(device=DEVICE,dtype=torch.float)
        #new_SA_img = new_SA_img.to(device=DEVICE,dtype=torch.float)

        with torch.no_grad():   
            
            out_LA = model(img_LA)    
            out_LA = generate_label_1(out_LA) # --- [B,C,H,W]
            
            print(out_LA.shape)

            out_LA_LV = out_LA[:,0:1,:,:]
            out_LA_MYO = out_LA[:,1:2,:,:]
            out_LA_RV = out_LA[:,2:3,:,:]
            BG = out_LA[:,3:4,:,:]
            
            Dice_score_LA_LV += (2 * (out_LA_LV * temp_LA[:,0:1,:,:].contiguous()).sum()) / (
                (out_LA_LV + temp_LA[:,0:1,:,:].contiguous()).sum() + 1e-8)
            
            Dice_score_LA_MYO += (2 * (out_LA_MYO*temp_LA[:,1:2,:,:].contiguous()).sum()) / (
    (out_LA_MYO + temp_LA[:,1:2,:,:].contiguous()).sum() + 1e-8)

            Dice_score_LA_RV += (2 * (out_LA_RV* temp_LA[:,2:3,:,:].contiguous()).sum()) / (
        (out_LA_RV + temp_LA[:,2:3,:,:].contiguous()).sum() + 1e-8)
        
            Dice_score_BG += (2 * (BG* temp_LA[:,3:4,:,:].contiguous()).sum()) / (
        (BG + temp_LA[:,3:4,:,:].contiguous()).sum() + 1e-8)
            
            
    print(f"Dice_score_LA_RV  : {Dice_score_LA_RV/len(loader)}")
    print(f"Dice_score_LA_MYO  : {Dice_score_LA_MYO/len(loader)}")
    print(f"Dice_score_LA_LV : {Dice_score_LA_LV/len(loader)}")
    print(f"Dice_score_BG : {Dice_score_BG/len(loader)}")
    
   
    Overall_Dicescore = Dice_score_LA_RV + Dice_score_LA_MYO + Dice_score_LA_LV + Dice_score_BG
    
    Overall_Dicescore = Overall_Dicescore/4
    
    return Overall_Dicescore/len(loader)

### 6 - This is Focal Tversky Loss loss function ### 

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='softmax'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='softmax'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))
           
        
## 7- This is the main Training function, where we will call all previous functions
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)

model_= Model_4()


loss_CORSS = nn.CrossEntropyLoss()

def main():
    model = model_.to(device=DEVICE,dtype=torch.float)

    loss_fn1 = DiceLoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader_ED,train_loader_ES,val_loader_ED,val_loader_ES, model, optimizer, loss_fn1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        dice_score_ED = check_Dice_Score(val_loader_ED, model, device=DEVICE)
        dice_score_ES = check_Dice_Score(val_loader_ES, model, device=DEVICE)
        
        dice_score = (dice_score_ED.detach().cpu().numpy() + dice_score_ES.detach().cpu().numpy())/2
        
        avg_valid_DS1.append(dice_score_ED.detach().cpu().numpy())
        avg_valid_DS2.append(dice_score_ES.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            break

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses1)+1),avg_train_losses1, label='Training Loss ED')
plt.plot(range(1,len(avg_valid_losses1)+1),avg_valid_losses1,label='Validation Loss ED')
plt.plot(range(1,len(avg_valid_DS1)+1),avg_valid_DS1,label='Validation DS ED')

plt.plot(range(1,len(avg_train_losses2)+1),avg_train_losses2, label='Training Loss ES')
plt.plot(range(1,len(avg_valid_losses2)+1),avg_valid_losses2,label='Validation Loss ES')
plt.plot(range(1,len(avg_valid_DS2)+1),avg_valid_DS2,label='Validation DS ES')

# find position of lowest validation loss
minposs = avg_valid_losses1.index(min(avg_valid_losses1))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses1)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
