import scipy
import scipy.ndimage
import nibabel as nib
import os
import SimpleITK as sitk


    ######### SA <---> LA #########
    
def LA_to_SA(SA_img,LA_img):
    # Get sizes
    SA_size = (SA_img.GetSize())
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

    
for i in range(10,100):
    path1 = r'C:\My_Data\M2M Data\data\data_2\train/'+ '0'+str(i)+'/0'+str(i)
    
    path = path1 + '_SA_ED.nii.gz'
    SA_img = sitk.ReadImage(path)
    
    path = path1 +'_LA_ED.nii.gz'
    LA_img = sitk.ReadImage(path)
    
    new_SA_img = LA_to_SA(SA_img,LA_img)
    new_LA_img = SA_to_LA(SA_img,LA_img)
    
    write_path = r'C:\My_Data\M2M Data\data\data_2\train/' + '0'+ str(i) +'/'
    

    sitk.WriteImage(new_SA_img, write_path+'0'+ str(i)+'_SA_ED_T.nii.gz')
    sitk.WriteImage(new_LA_img, write_path+'0'+ str(i) +'_LA_ED_T.nii.gz')


