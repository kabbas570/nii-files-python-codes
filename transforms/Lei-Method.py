
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os 
import nibabel as nib

SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\001\001_SA_ES.nii.gz')  ### path to SA image 
LA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\001\001_LA_ES.nii.gz')    ## path to LA image 

#LA_img = sitk.GetArrayFromImage(LA_img)
#SA_img = sitk.GetArrayFromImage(SA_img)

#print(LA_img.shape)
# LA_img = np.repeat(LA_img, 12, axis=0)

# print(SA_img.shape)
# print(LA_img.shape)

# LA_img = sitk.GetImageFromArray(LA_img)
# SA_img = sitk.GetImageFromArray(SA_img)

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
            
t1 = sitk.GetArrayFromImage(new_img)

save_1 = r'C:\My_Data\M2M Data\save2'
for i in range(12):
    plt.figure()
    t1 = np.transpose(t1, (0,2,1))
    plt.imshow(t1[i,:,:])
    #plt.imsave(os.path.join(save_1,str(i)+".png"),t1[i,:,:])
