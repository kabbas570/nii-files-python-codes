import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt



def bi_project_dataV2(source, target):
    # print(f"{source} {target}")
    target_img=target
    source_img=source
    # source_img=sitkResample3DV2(source_img,sitk.sitkLinear,target_img.GetSpacing())

    new_img=sitk.Image(target_img.GetSize(),source_img.GetPixelID())
    new_img.CopyInformation(target_img)
    size=(source_img.GetSize())

    print(size)
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            for z in range(0,size[2]):

                point=source_img.TransformIndexToPhysicalPoint([x,y,z])
                # p=index2physicalpoint(target_img,[x,y,z])
                # index_la=la_img.TransformPhysicalPointToContinuousIndex(point)
                index_la=target_img.TransformPhysicalPointToIndex(point)
                # index_la=np.round(index_la)
                # i=physicalpoint2index(source_img, point)
                index_la=np.array(index_la)
                if index_la[0]<0 or index_la[0] >= target_img.GetSize()[0]:
                    continue
                if index_la[1] < 0 or index_la[1] >= target_img.GetSize()[1]:
                    continue
                if index_la[2] < 0 or index_la[2] >= target_img.GetSize()[2]:
                    continue
                new_img[int(index_la[0]),int(index_la[1]),int(index_la[2])]=0
                # print(index_la)
                # print(x,y,z)
                # new_img[x,y,z]=source_img.GetPixel(x,y,z)

                new_img[int(index_la[0]),int(index_la[1]),int(index_la[2])]=source_img[x,y,z]
                # new_img[x,y,z]=interplote(la_img,index_la)
    return new_img


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




SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\005\005_SA_ES_gt.nii.gz')  ### path to SA image 
LA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\train\005\005_LA_ES_gt.nii.gz')    ## path to LA image 




proj = LA_to_SA(SA_img,LA_img)

#proj = bi_project_dataV2(LA_img,SA_img)

#proj = SA_to_LA(SA_img,LA_img)

proj=sitk.GetArrayFromImage(proj)

SA_img=sitk.GetArrayFromImage(SA_img)



for i in range(5,6):
    plt.figure()
    plt.imshow(SA_img[i,:])
    
for i in range(4,5):
    plt.figure()
    plt.imshow(proj[i,:])
    
    
import cv2

def give_centers(T):
    target_centers=[]
    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])  
        cY = int(M["m01"] / M["m00"])
        target_centers.append((cX,cY))
    return target_centers


temp = np.zeros([256,256])

img = proj[4,:]
temp[np.where(img==1)]=1

#centers = give_centers(temp)

def draw_circle(T):
    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
    img = cv2.circle(T, center, radius=radius, color=1, thickness=-1)
    
    return img

def draw_semi_circle(T):
    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
    image = cv2.ellipse(T, center, (radius,radius), angle=90, startAngle=0, endAngle=180, color=1,thickness=-1)

    
    return image

temp = np.zeros([256,256])

img = proj[4,:]
temp[np.where(img==2)]=1

#result = draw_circle(temp)





def draw_circle_myo(T,myo):
    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius1 = int(radius)
    
    T = myo
    T = T.astype(np.uint8)
    cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    radius = int(radius)
    
    image = cv2.circle(T, center, radius=radius1+radius, color=2, thickness=radius)
    
    return image



img = proj[4,:]
temp = np.zeros([256,256])
temp[np.where(img==1)]=1
    
temp1 = np.zeros([256,256])
temp1[np.where(img==2)]=1

myo = draw_circle_myo(temp,temp1)



a = draw_circle(temp)


two = np.zeros([256,256])
two[np.where(myo==2)]=2
two[np.where(a==1)]=1


