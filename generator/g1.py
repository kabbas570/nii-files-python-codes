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
        
        
        temp_SA=np.zeros([3,org_dim3,DIM_,DIM_])
        temp_SA[0,:,:,:][np.where(img_SA_gt==1)]=1
        temp_SA[1,:,:,:][np.where(img_SA_gt==2)]=1
        temp_SA[2,:,:,:][np.where(img_SA_gt==3)]=1
   
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
        
        
        temp_LA = np.zeros([3,org_dim3,DIM_,DIM_])
        temp_LA[0,:,:,:][np.where(img_LA_gt==1)]=1
        temp_LA[1,:,:,:][np.where(img_LA_gt==2)]=1
        temp_LA[2,:,:,:][np.where(img_LA_gt==3)]=1
        
        ### sa_to_la mapping ####
        
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ES.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ES.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = LA_to_SA(img_SA_1,img_LA_1)
        new_SA_img = resample_image_SA(new_SA_img)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        

        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 

        new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
        
        new_SA_img = Normalization_1(new_SA_img)
        new_SA_img = Normalization_2(new_SA_img)
        new_SA_img = np.expand_dims(new_SA_img, axis=0)

        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,new_SA_img,self.images_name[index]
           
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
        
        
        temp_SA=np.zeros([3,org_dim3,DIM_,DIM_])
        temp_SA[0,:,:,:][np.where(img_SA_gt==1)]=1
        temp_SA[1,:,:,:][np.where(img_SA_gt==2)]=1
        temp_SA[2,:,:,:][np.where(img_SA_gt==3)]=1
   
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
        
        
        temp_LA = np.zeros([3,org_dim3,DIM_,DIM_])
        temp_LA[0,:,:,:][np.where(img_LA_gt==1)]=1
        temp_LA[1,:,:,:][np.where(img_LA_gt==2)]=1
        temp_LA[2,:,:,:][np.where(img_LA_gt==3)]=1
        
        ### sa_to_la mapping ####
        
        img_path_SA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_SA_path = img_path_SA +'_SA_ED.nii.gz'
        img_SA_1 = sitk.ReadImage(img_SA_path)
        img_path_LA = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path = img_path_LA +'_LA_ED.nii.gz'
        img_LA_1 = sitk.ReadImage(img_LA_path)
        
        new_SA_img = LA_to_SA(img_SA_1,img_LA_1)
        new_SA_img = resample_image_SA(new_SA_img)
        new_SA_img = sitk.GetArrayFromImage(new_SA_img)
        

        org_dim3 = new_SA_img.shape[0]
        org_dim1 = new_SA_img.shape[1]
        org_dim2 = new_SA_img.shape[2] 

        new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)
        
        new_SA_img = Normalization_1(new_SA_img)
        new_SA_img = Normalization_2(new_SA_img)
        new_SA_img = np.expand_dims(new_SA_img, axis=0)

        return img_LA,temp_LA[:,0,:,:],img_SA,temp_SA,new_SA_img,self.images_name[index]
           
def Data_Loader_Both_ED(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_Both_ED(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


train_imgs = r'C:\My_Data\M2M Data\data\data_2\train' 
train_csv_path = r'C:\My_Data\M2M Data\data\train.csv' 
df_train = pd.read_csv(train_csv_path)
train_loader_ES = Data_Loader_Both_ES(df_train,train_imgs,batch_size = 1)
a = iter(train_loader_ES)
a1 = next(a)

# for i in range(6):
#     gt =  a1[4][0,0,:,:,:]
#     label = a1[5].numpy()
#     plt.figure()
#     plt.imshow(gt[i,:,:])

# SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\084\084_LA_ES_gt.nii.gz')
# SA_img = resample_image_LA(SA_img)
# t1 = sitk.GetArrayFromImage(SA_img)

# plt.figure()
# plt.imshow(t1[0,:,:])

# org_dim3 = t1.shape[0]
# org_dim1 = t1.shape[1]
# org_dim2 = t1.shape[2] 

# new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,256,t1)

# plt.figure()
# plt.imshow(new_SA_img[0,:,:])

# r =np.zeros([1,256,256])

# r[np.where(new_SA_img==1)]=1

# plt.figure()
# plt.imshow(r[0,:,:])


# SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\084\084_SA_ES.nii.gz')
# LA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\084\084_LA_ES.nii.gz')

# b1 = LA_to_SA(SA_img,LA_img)

# new_SA_img = resample_image_SA(b1)
# new_SA_img = sitk.GetArrayFromImage(new_SA_img)


# org_dim3 = new_SA_img.shape[0]
# org_dim1 = new_SA_img.shape[1]
# org_dim2 = new_SA_img.shape[2] 

# new_SA_img = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,new_SA_img)

# new_SA_img = Normalization_1(new_SA_img)
# new_SA_img = np.expand_dims(new_SA_img, axis=0)

# for i in range(2):
#     plt.figure()
#     plt.imshow(new_SA_img[0,i,:,:])
