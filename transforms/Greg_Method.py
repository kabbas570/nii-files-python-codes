import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

# Load images
SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\134\134_SA_ES.nii.gz')
LA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\134\134_LA_ES.nii.gz')


def LA_to_SA(SA_img,LA_img):
    # Get sizes
    SA_size = (SA_img.GetSize())
    LA_size = (LA_img.GetSize())
    
    print(SA_size)
    print(LA_size)
    
    
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

new_SA_img = LA_to_SA (SA_img,LA_img)
print(new_SA_img.GetSize())
newSA = sitk.GetArrayFromImage(new_SA_img)
print(newSA.shape)

# Visualise and save new short axis image
for i in range(3):
    plt.figure()
    #temp = np.transpose(newSA, (0, 2, 1))
    plt.imshow(newSA[i, :, :])
    #plt.imsave('D:\\Greg\\Research\\Code\\abbas\\001\\001_sa' + str(i) + '.png',temp[i, :, :])

# Visualise and save original short axis -- the new SA and original SA should match where they overlap
origSA = sitk.GetArrayFromImage(SA_img)
for i in range(12):
    plt.figure()
    temp = np.transpose(origSA, (0, 2, 1))
    plt.imshow(temp[i, :, :])
    #plt.imsave('D:\\Greg\\Research\\Code\\abbas\\001\\001_Orig_sa' + str(i) + '.png', temp[i, :, :])
