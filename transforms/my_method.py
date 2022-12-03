import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import scipy
import scipy.ndimage


    ######### SA <---> LA #########
modality_SA = nib.load(r'C:\My_Data\M2M Data\data\train\001\001_SA_ES.nii.gz')
modality_LA = nib.load(r'C:\My_Data\M2M Data\data\train\001\001_LA_ES.nii.gz')

aff_SA = modality_SA.affine
aff_LA = modality_LA.affine
                                          ##  for SA ##
#transformed_img = scipy.ndimage.affine_transform(modality_SA.get_fdata(),(np.linalg.inv(aff_LA)) @ (aff_SA))
transformed_img = scipy.ndimage.affine_transform(modality_SA.get_fdata(),(np.linalg.inv(aff_SA)) @ (aff_LA))
                                             ##  for LA ##
#transformed_img = scipy.ndimage.affine_transform(modality_SA.get_fdata(),(np.linalg.inv(aff_LA)) @ aff_SA)

#transformed_img = scipy.ndimage.affine_transform(modality_LA.get_fdata(),(np.linalg.inv(aff_LA)) @ (aff_SA))

for i in range(12):
    plt.figure()
    plt.imshow(transformed_img[:,:,i])