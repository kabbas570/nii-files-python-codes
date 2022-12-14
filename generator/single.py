import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from typing import List, Union, Tuple
import torch
import numpy as np
from scipy import ndimage

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset


k_folds = 4
kfold = KFold(n_splits=k_folds, shuffle=True)

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256


def Generate_Meta_Data_vendor(vendors_):
   
    if vendors_=='GE MEDICAL SYSTEMS':
        V = np.array([1,1,1])
    if vendors_=='SIEMENS':
        V = np.array([1,1,0])
    if vendors_=='Philips Medical Systems':
        V = np.array([1,0,0])  
    V = np.expand_dims(V, axis=0)
    return V


def Generate_Meta_Data_scanner(scanners_):
        if scanners_=='Symphony':
            S = np.array([1,1,1,1,1,1,1,1,1])
        if scanners_=='SIGNA EXCITE':
            S = np.array([1,1,1,1,1,1,1,1,0])
        if scanners_=='Signa Explorer':
            S = np.array([1,1,1,1,1,1,1,0,0])
        if scanners_=='SymphonyTim':
            S = np.array([1,1,1,1,1,1,0,0,0])
        if scanners_=='Avanto Fit':
            S = np.array([1,1,1,1,1,0,0,0,0])
        if scanners_=='Avanto':
            S = np.array([1,1,1,1,0,0,0,0,0])
        if scanners_=='Achieva':
            S = np.array([1,1,1,0,0,0,0,0,0])
        if scanners_=='Signa HDxt':
            S = np.array([1,1,0,0,0,0,0,0,0])
        if scanners_=='TrioTim':
            S = np.array([1,0,0,0,0,0,0,0,0])
        S=np.expand_dims(S, axis=0)
        S=np.expand_dims(S, axis=0)
        return S
        
def Generate_Meta_Data_disease(diseases_):  
        if diseases_=='FALL':
            D = np.array([1,1,1,1,1,1,1,1])
        if diseases_=='RV':
            D = np.array([1,1,1,1,1,1,1,0])
        if diseases_=='CIA':
            D = np.array([1,1,1,1,1,1,0,0])
        if diseases_=='ARR':
            D = np.array([1,1,1,1,1,0,0,0])
        if diseases_=='TRI':
            D = np.array([1,1,1,1,0,0,0,0])
        if diseases_=='LV':
            D = np.array([1,1,1,0,0,0,0,0])
        if diseases_=='NOR':
            D = np.array([1,1,0,0,0,0,0,0])
        if diseases_=='HCM':
            D = np.array([1,0,0,0,0,0,0,0])
        D=np.expand_dims(D, axis=0)
        D=np.expand_dims(D, axis=0)
        return D

def same_depth(img):
    temp = np.zeros([img.shape[0],17,256,256])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
    
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

def generate_label_1(gt,org_dim3):
        temp_ = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        temp_[3,:,:,:][np.where(gt==0)]=1
        return temp_
        
def generate_label_2(gt,org_dim3):
    
        temp_ = np.zeros([6,org_dim3,DIM_,DIM_])
    
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[1,:,:,:][np.where(gt==3)]=1
        
        temp_[2,:,:,:][np.where(gt==2)]=1
        temp_[3,:,:,:][np.where(gt==1)]=1
        temp_[3,:,:,:][np.where(gt==3)]=1
        
        temp_[4,:,:,:][np.where(gt==3)]=1
        temp_[5,:,:,:][np.where(gt==2)]=1
        temp_[5,:,:,:][np.where(gt==1)]=1

        return temp_

class Dataset_(Dataset): 
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
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        temp_SA_ES = generate_label_1(img_SA_gt,org_dim3)        
        temp_SA_ES = same_depth(temp_SA_ES)
        
   
        #####    LA Images #####
        ## la_es_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ES = Normalization_1(img_LA)

        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ES = generate_label_1(img_LA_gt,org_dim3)
        
        ## ED images ##
        ## sa_eD_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)
        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        temp_SA_ED = generate_label_1(img_SA_gt,org_dim3)        
        temp_SA_ED = same_depth(temp_SA_ED)

        #####    LA Images #####
        ## la_ed_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ED = Normalization_1(img_LA)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ED = generate_label_1(img_LA_gt,org_dim3)

        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        V = Generate_Meta_Data_vendor(vendors_)
        S = Generate_Meta_Data_scanner(scanners_)
        D = Generate_Meta_Data_disease(diseases_)
        
        return img_LA_ES,temp_LA_ES[:,0,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,0,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],V,S,D

def Data_Loader_(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# val_imgs= r'C:\My_Data\M2M Data\data\val'
# val_csv_path= r'C:\My_Data\M2M Data\data\val.csv'
# df_val = pd.read_csv(val_csv_path)
# train_loader_ED = Data_Loader_(df_val,val_imgs,batch_size=1)
# a = iter(train_loader_ED)
# a1 =next(a)
# plt.figure()
# plt.imshow(a1[3][0,1,4,:,:])
# plt.figure()
# plt.imshow(a1[7][0,1,4,:,:])
# print(a1[8][0])
# img_path = r'C:\My_Data\M2M Data\data\data_2\val\199\199'
# img_ED_path = img_path+'_SA_ED_gt.nii.gz'
# img_LA = sitk.ReadImage(img_ED_path)
# img_LA = resample_image_LA(img_LA)
# img_LA = sitk.GetArrayFromImage(img_LA)
# org_dim3 = img_LA.shape[0]
# org_dim1 = img_LA.shape[1]
# org_dim2 = img_LA.shape[2] 
# img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
# img_LA_ED = Normalization_1(img_LA)
# plt.figure()
# plt.imshow(img_LA_ED[4,:,:])


    
    
    
    
