import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd

import nibabel.processing
import nibabel as nib



img_=r'C:\My_Data\M2M Data\data\train\001/001'+'_SA_ES.nii.gz'
img_ = nib.load(img_)

n1_header = img_.header
print(n1_header) 

img_.header['pixdim'] = [1,1,1,1,1,1,1,1] ## set the voxel dim here

affine = img_.affine
img = nib.Nifti1Image(img_, affine, n1_header)
n2_header = img.header
print(n2_header) 


sx, sy, sz = img_.header.get_zooms()
volume = sx * sy * sz
print(volume)

sx, sy, sz = img.header.get_zooms()
volume = sx * sy * sz
print(volume)
