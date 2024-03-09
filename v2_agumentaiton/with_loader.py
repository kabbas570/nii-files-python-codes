import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import albumentations as A
import torchvision 

NUM_WORKERS=0
PIN_MEMORY=True

from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

transform_geo1 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.7),
    v2.RandomVerticalFlip(p=0.7),
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

class Dataset_train(Dataset):
    def __init__(self, image_dir, mask_rust, mask_st):
        self.image_dir = image_dir
        self.mask_rust = mask_rust
        self.mask_st = mask_st
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_rust = os.path.join(self.mask_rust, self.images[index])
        mask_st = os.path.join(self.mask_st, self.images[index])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image,-1,0)
        
        st_img = cv2.imread(mask_st)
        st_img = cv2.cvtColor(st_img, cv2.COLOR_BGR2RGB)
        st_img = np.moveaxis(st_img,-1,0)
       
       
        rust_gt = cv2.imread(mask_rust)
        rust_gt = cv2.cvtColor(rust_gt, cv2.COLOR_BGR2RGB)
        rust_gt = np.moveaxis(rust_gt,-1,0)
        
        ## apply augmentaitons ##
        all_three = np.stack((image,st_img,rust_gt), axis=0)
        all_three = torch.tensor(all_three)
        transformed_images = transforms_geo(all_three)
        
        # Get the transformed images:
        image = transformed_images[0]
        st_img = transformed_images[1]
        rust_gt = transformed_images[2]

        image = transforms_inten(image)
        image = image/255
        
        st_img = st_img[0:1,:]
        rust_gt = rust_gt[0:1,:]

        st_img = st_img / 255
        st_img = (st_img >= 0.5)*1
        
        rust_gt = (rust_gt >= 0.5)*1
        
        
        return image,st_img,rust_gt, self.images[index][:-4]

def Data_Loader_train(image_dir,mask_rust,mask_st,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_train( image_dir=image_dir, mask_rust=mask_rust, mask_st=mask_st)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


# train_imgs =  r'C:\My_Data\KeenAI\rust_new\DEC_15\val\imgs/'
# train_rust = r'C:\My_Data\KeenAI\rust_new\DEC_15\val\gts/'
# train_st = r'C:\My_Data\KeenAI\rust_new\DEC_15\val\steel/'

# val_loader = Data_Loader_train(train_imgs,train_rust,train_st,1)
# print(len(val_loader))   
# a = iter(val_loader)
# a1 = next(a)

# img = a1[0][0,:].numpy()
# st = a1[1][0,:].numpy()
# gt =  a1[2][0,:].numpy()

# img = np.moveaxis(img,0,2)
# plt.figure()
# plt.imshow(img)

# plt.figure()
# plt.imshow(gt[0,:])

# plt.figure()
# plt.imshow(st[0,:])
