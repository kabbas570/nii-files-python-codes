
import SimpleITK as sitk
from typing import List, Union, Tuple

from multiprocessing import Pool

import numpy as np
from scipy import ndimage

def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (1.0, 1.0, 1.0),
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
    

import nibabel.processing
import nibabel as nib
img_SA_path=r'C:\My_Data\M2M Data\data\train\001/001'+'_LA_ES.nii.gz'
img_LA_gt =nib.load(img_SA_path)

SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\001\001_SA_ES.nii.gz')

original_spacing = np.array(SA_img.GetSpacing())

a=resample_image(SA_img)

original_spacing = np.array(a.GetSpacing())


t1 = sitk.GetArrayFromImage(a)




