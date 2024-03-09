import torch
import torchvision 
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

transform_geo1 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(0, 60)),
])

transform_geo2 = v2.Compose([
    v2.ElasticTransform(alpha=300.0),
    v2.RandomResizedCrop(size=(224, 224), antialias=True)
    
])

transforms_geo =  v2.RandomChoice([
    transform_geo1,
    transform_geo2,
    v2.Compose([])  # No transform
], p=[0.3, 0.2, 0.5])

transform_inten1 = v2.RandomChoice([
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    v2.RandomAdjustSharpness(sharpness_factor=4),
    v2.RandomEqualize(),
],p=[0.3, 0.4, 0.3])

transforms_inten =  v2.RandomChoice([
    transform_inten1,
    v2.Compose([])  # No transform
], p=[0.5, 0.5])


import cv2
import numpy as np
import matplotlib.pyplot as plt

path_img = r"C:\My_Data\KeenAI\rust_new\DEC_15\val\imgs\5HCIwWB8zye2BJ6hmj9RA==_1701.png"
path_st_img = r"C:\My_Data\KeenAI\rust_new\DEC_15\val\steel\5HCIwWB8zye2BJ6hmj9RA==_1701.png"
path_label = r"C:\My_Data\KeenAI\rust_new\DEC_15\val\gts\5HCIwWB8zye2BJ6hmj9RA==_1701.png"


img = cv2.imread(path_img)
st_img = cv2.imread(path_st_img)
label = cv2.imread(path_label)*255

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(st_img)

plt.figure()
plt.imshow(label)


img = np.moveaxis(img,-1,0)
st_img = np.moveaxis(st_img,-1,0)
label = np.moveaxis(label,-1,0)

all_three = np.stack((img,st_img,label), axis=0)

# Apply the transformations to both images simultaneously:

all_three = torch.tensor(all_three)
#transformed_images = T.RandomRotation(180)(all_three)
transformed_images = transforms_geo(all_three)



# Get the transformed images:
img = transformed_images[0]
img_st = transformed_images[1]
label = transformed_images[2]

img = transforms_inten(img)

img = torch.moveaxis(img,0,-1)
img_st = torch.moveaxis(img_st,0,-1)
label = torch.moveaxis(label,0,-1)

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(img_st)

plt.figure()
plt.imshow(label)
