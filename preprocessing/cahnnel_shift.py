import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

# Load images
SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\val\200\200_SA_ES.nii.gz')

SA_img = sitk.GetArrayFromImage(SA_img)
print(SA_img.shape)

new = np.transpose(SA_img, (1,2,0))  ## to bring channel as last dimenssion 

for i in range(3,6):
    plt.figure()
    plt.imshow(new[:,:,i])
    
for i in range(3,6):
    plt.figure()
    plt.imshow(SA_img[i,:,:])
    
    
    
img_new = np.transpose(new, (2,0,1))  ## to bring channel first 
    
for i in range(3,6):
    plt.figure()
    plt.imshow(img_new[i,:,:])  
