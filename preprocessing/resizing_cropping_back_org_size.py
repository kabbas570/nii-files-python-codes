import torch
import torch.nn as nn

class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,kernel_size):
        super().__init__()
        
        self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU()
        )

    def forward(self, x1):
        x = self.up(x1)
        return  x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)
class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)
            
Base = 64
class Autoencoder_2D2_1(nn.Module):
    def __init__(self, n_channels = 3, bilinear=False):
        super(Autoencoder_2D2_1, self).__init__()
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            
        Conv(n_channels, Base,1,1,3),
        Conv(Base, Base,1,1,3),
        Conv(Base, Base,2,1,3),
        
        Conv(Base, 2*Base,1,1,3),
        Conv(2*Base, 2*Base,1,1,3),
        Conv(2*Base, 2*Base,2,1,3),
        
        Conv(2*Base,4*Base,1,1,3), 
        Conv(4*Base,4*Base,1,1,3), 
        Conv(4*Base, 4*Base,2,1,3), 
        
        Conv(4*Base,8*Base,1,1,3), 
        Conv(8*Base,8*Base,1,1,3), 
        Conv(8*Base, 8*Base,2,1,3), 

        )
        
        self.decoder =  nn.Sequential(
            
        Deconv(8*Base,8*Base,2,2),
        Conv(8*Base,8*Base,1,1,3),
        Conv(8*Base,8*Base,1,1,3),
        
        Deconv(8*Base,4*Base,2,2),
        Conv(4*Base,4*Base,1,1,3),
        Conv(4*Base,4*Base,1,1,3),
        
        Deconv(4*Base,2*Base,2,2),
        Conv(2*Base,2*Base,1,1,3),
        Conv(2*Base,2*Base,1,1,3),
        
        Deconv(2*Base,Base,2,2),
        Conv(Base,Base,1,1,3),
        Conv(Base,Base,1,1,3),
        
        OutConv_2d(Base,3),
        )

    def forward(self, x_in):
      x = self.encoder(x_in)
      encoded = self.decoder(x)
      return encoded

# Input_Image_Channels = 3
# def model() -> Autoencoder_2D2_1:
#     model = Autoencoder_2D2_1()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple
import torch
import cv2
from torch.utils.data import SubsetRandomSampler
import torchio as tio
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256

# def SDF_2D(img):
#     img = img.astype(np.uint8)
#     normalized_sdf = np.zeros(img.shape)
#     posmask = img.astype(bool)
#     if posmask.any():
#         negmask = ~posmask
#         posdis = distance(posmask)
#         negdis = distance(negmask)
#         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#         sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))*negmask - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))*posmask
#         sdf[boundary==1] = 0
#         normalized_sdf = sdf
#     return normalized_sdf
 

# def SDF_3D(img_gt):
#     img_gt = img_gt.astype(np.uint8)
#     normalized_sdf = np.zeros(img_gt.shape)
#     posmask = img_gt.astype(bool)
#     if posmask.any():
#         negmask = ~posmask
#         posdis = distance(posmask)
#         negdis = distance(negmask)
#         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#         sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
#         sdf[boundary==1] = 0
#         normalized_sdf = sdf
#         assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
#         assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

#     return normalized_sdf

    
def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
    
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
    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
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
   
def generate_label_3(gt,org_dim3):
        temp_ = np.zeros([3,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        return temp_


def generate_label_4(gt,org_dim3):
        temp_ = np.zeros([1,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[0,:,:,:][np.where(gt==2)]=1
        temp_[0,:,:,:][np.where(gt==3)]=1
        return temp_

transforms_all = tio.OneOf({
        tio.RandomBiasField(): .3,  ## axis [0,1] or [1,2]
        #tio.RandomGhosting(axes=([1,2])): 0.3,
        tio.RandomFlip(axes=([1,2])): .3,  ## axis [0,1] or [1,2]
        tio.RandomFlip(axes=([0,1])): .3,  ## axis [0,1] or [1,2]
        tio.RandomAffine(degrees=(30,0,0)): 0.3, ## for 2D rotation 
        #tio.RandomMotion(degrees =(30) ):0.3 ,
        #tio.RandomBlur(): 0.3,
        #tio.RandomGamma(): 0.3,   
        #tio.RandomNoise(mean=0.1,std=0.1):0.20,
})

def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
def Normalization_LA_ED(img):
        img = (img-114.7321)/189.8573
        return img 
        
def Normalization_SA_ES(img):
        img = (img-62.5983)/147.4826
        return img 
def Normalization_SA_ED(img):
        img = (img-62.9529)/147.6579
        return img 

def SA_to_LA(SA_img,LA_img):    ## new LA image 
    size = (LA_img.GetSize()) 
    new_img = sitk.Image(LA_img)
    
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            for z in range(0,size[2]):
                new_img[x,y,z] = 0
                point = LA_img.TransformIndexToPhysicalPoint([x,y,z])  ##  determine the physical location of a pixel:
                index_LA = SA_img.TransformPhysicalPointToIndex(point)
                if index_LA[0] < 0 or index_LA[0]>= SA_img.GetSize()[0]:
                    continue
                if index_LA[1] < 0 or index_LA[1]>= SA_img.GetSize()[1]:
                    continue
                if index_LA[2] < 0 or index_LA[2]>= SA_img.GetSize()[2]:
                    continue
                new_img[x,y,z] = SA_img[index_LA[0],index_LA[1],index_LA[2]]
    
    return new_img
    
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ES(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt_org = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt_org) 
        temp_SA_ES = np.expand_dims(img_SA_gt, axis=0)
        
        print(self.images_name[index])
        
        temp_SA_ES = temp_SA_ES[0,:]
        return img_SA_gt_org,temp_SA_ES,img_SA_gt_org.shape
        

def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r"C:\My_Data\M2M Data\data\data_2\single\imgs"
val_csv_path = r"C:\My_Data\M2M Data\data\data_2\single\val1.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
#train_loader = Data_Loader_V(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)
#a1 = next(a)



for i in range(1):
    a1 = next(a)
    
    org = a1[0][0,:].numpy()
    cropped = a1[1][0,:].numpy()
    org_shape = a1[2]
    
    DIM_ = 256
    def Back_to_orgSize(img,org_shape):
        if org_shape[1]>DIM_ and org_shape[2]>DIM_:
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            d2 = int((org_shape[2]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            n2 = int(org_shape[2]-d2)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,d1:n1,d2:n2] = img
            
          
        if org_shape[1]<DIM_ and org_shape[2]<DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)
            d2 = int((DIM_-org_shape[2])/2)
            n1 = int(DIM_-d1)
            n2 = int(DIM_-d2)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp = img[:,d1:n1,d2:n2]
            
        if org_shape[1]>DIM_ and org_shape[2]==DIM_:
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            temp[:,d1:n1,:] = img
            
        if org_shape[1]==DIM_ and org_shape[2]==DIM_:
            temp = img
            
        if org_shape[1]==DIM_ and org_shape[2]<DIM_:
            temp = np.zeros(org_shape)
            d2 = int((DIM_-org_shape[2])/2)
            n2 = int(DIM_-d2)
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp = img[:,:,d2:n2]
            

        if org_shape[1]>DIM_ and org_shape[2]<DIM_:
            
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
                
                ## for smaller side 
            d2 = int((DIM_-org_shape[2])/2)
            n2 = int(DIM_-d2)
            
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,d1:n1,:] = img[:,:,d2:n2]
            
        if org_shape[1]<DIM_ and org_shape[2]>DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)
            n1 = int(DIM_-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
           
            d2 = int((org_shape[2]-DIM_)/2)
            n2 = int(org_shape[2]-d2)
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,:,d2:n2] = img[:,d1:n1,:]

        if org_shape[1]<DIM_ and org_shape[2]==DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)     
            n1 = int(DIM_-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
    
            temp = img[:,d1:n1,:]
            
        return temp
    
    back = Back_to_orgSize(cropped,org_shape)
    
    # plt.figure()
    # plt.imshow(back[0,:])
    
    # plt.figure()
    # plt.imshow(org[0,:])
    
    s1 = np.array_equal(back, org)
    print(int(s1))


b1 = back[1,:]
plt.figure()
plt.imshow(b1)

o1 = org[1,:]
plt.figure()
plt.imshow(o1)


for i in range(4):
    s1 = np.array_equal(back[i,:], org[i,:])
    print(int(s1))
    
    b1 = back[i,:]
    plt.figure()
    plt.imshow(b1)

    o1 = org[i,:]
    plt.figure()
    plt.imshow(o1)
    
    
back_label = generate_label_3(back, 17)
org_label = generate_label_3(org, 17)

s1 = np.array_equal(back_label,org_label)
print(int(s1))
