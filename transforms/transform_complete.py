import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def bi_project_dataV2(source, target):
    # print(f"{source} {target}")
    target_img=target
    source_img=source
    # source_img=sitkResample3DV2(source_img,sitk.sitkLinear,target_img.GetSpacing())

    new_img=sitk.Image(target_img.GetSize(),source_img.GetPixelID())
    new_img.CopyInformation(target_img)
    size=(source_img.GetSize())
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


SA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\data_2/val\166\166_SA_ES_gt.nii.gz')  ### path to SA image 
LA_img = sitk.ReadImage(r'C:\My_Data\M2M Data\data\data_2/val\166\166_LA_ES_gt.nii.gz')    ## path to LA image 
proj = bi_project_dataV2(LA_img,SA_img)

proj=sitk.GetArrayFromImage(proj)

DIM_ = 256
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


org_dim3 = proj.shape[0]
org_dim1 = proj.shape[1]
org_dim2 = proj.shape[2] 
        
proj = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,proj)

def draw_RV(rv,LV_MYO):
    rv = rv.astype(np.uint8)
    cnts = cv2.findContours(rv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    
    rows,cols = rv.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(rv,(cols-1,righty),(0,lefty),(0,255,0),2)

    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    
   # myradians = math.atan2(center[1]-(center[1]-radius), center[0]-(center[0]-radius))
    
    myradians = math.atan2(center[1]-(center[1]-int(radius/2)), center[0]-(center[0]-int(radius/2)))
    
    mydegrees = math.degrees(myradians)
    cv2.ellipse(LV_MYO, center, (radius,2*radius), angle=180-mydegrees, startAngle=0, endAngle=180, color=3,thickness=-1)
    cv2.ellipse(LV_MYO, center, (radius,2*radius), angle=270+mydegrees, startAngle=0, endAngle=180, color=3,thickness=-1)
    return LV_MYO

def fill_(temp):
    temp = temp.astype(np.uint8)
    cnts = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    cv2.drawContours(temp,cnt,-1,(1,1,1),2)
    kernel = np.ones((7,7),np.uint8)
    T = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    return T


def draw_LV(LV,myo,rv):
    LV = fill_(LV)
    myo = fill_(myo)
    rv = fill_(rv)
    
    cnts = cv2.findContours(LV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius_lv = int(radius)
    LV = cv2.circle(LV, center, radius=radius_lv, color=1, thickness=-1)


    cnts = cv2.findContours(myo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radii = []
    for k in range(len(cnts)):
        cnt = cnts[0][k]
        (_,_),radius = cv2.minEnclosingCircle(cnt)
        radii.append(radius)
    max_radius = max(radii)
    radius_myo = int(max_radius)
    LV_MYO = cv2.circle(LV, center, radius=radius_lv+radius_myo, color=2, thickness=2*radius_myo)
    
    x = draw_RV(rv,LV_MYO)
    
    return x

def myo_first(LV,myo,rv):
    LV = fill_(LV)
    myo = fill_(myo)
    rv = fill_(rv)
    
    
    plt.figure()
    plt.imshow(myo)
        
    
    cnts = cv2.findContours(LV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0][0]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius_lv = int(radius)
    LV = cv2.circle(LV, center, radius=radius_lv, color=1, thickness=-1)


    cnts = cv2.findContours(myo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radii = []
    print(len(cnts))
    for k in range(len(cnts[0])):
        cnt = cnts[0][k]
        (_,_),radius = cv2.minEnclosingCircle(cnt)
        radii.append(radius)
    max_radius = max(radii)
    radius_myo = int(max_radius)
    LV_MYO = cv2.circle(LV, center, radius=radius_lv+radius_myo, color=2, thickness=2*radius_myo)
    
    LV_MYO = cv2.circle(LV_MYO, center, radius=radius_lv, color=1, thickness=-1)
    
    x = draw_RV(rv,LV_MYO)
    
    return x



img_lv = np.zeros([10,256,256])
img_lv[np.where(proj==1)]=1

img_myo = np.zeros([10,256,256])
img_myo[np.where(proj==2)]=1

img_rv = np.zeros([10,256,256])
img_rv[np.where(proj==3)]=1

LV_3d = np.zeros([10,256,256])

for i in range(9,10):
    if np.sum(img_lv[i,:])!=0 and np.sum(img_myo[i,:])!=0 and np.sum(img_rv[i,:])!=0:
        
        # plt.figure()
        # plt.imshow(img_lv[i,:])
        
        plt.figure()
        plt.imshow(img_myo[i,:])
        
        # plt.figure()
        # plt.imshow(img_rv[i,:])
        
        print(i)
        a = myo_first(img_lv[i,:],img_myo[i,:],img_rv[i,:])
        #a_rv = draw_RV(img_rv[i,:])
        
        LV_3d[i,:] = a
        #LV_3d[i,:] = a_rv
        



i = 9
a = myo_first(img_lv[i,:],img_myo[i,:],img_rv[i,:])


s =img_myo[9,:]
plt.figure()
plt.imshow(s)

myo = fill_(s)

cnts = cv2.findContours(myo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(cnts[0][0])
radii = []
print(len(cnts[0]))
for k in range(len(cnts)):
    cnt = cnts[0][k]
    (_,_),radius = cv2.minEnclosingCircle(cnt)
    radii.append(radius)
max_radius = max(radii)
radius_myo = int(max_radius)


plt.figure()
plt.imshow(myo)

s =proj[9,:]
plt.figure()
plt.imshow(s)





SA_img=sitk.GetArrayFromImage(SA_img)

for i in range(9,10):
    plt.figure()
    plt.imshow(SA_img[i,:])

for i in range(8,9):
    plt.figure()
    plt.imshow(proj[i,:])




# g = LV_3d[4,:]
# g[np.where(proj[4,:]==1)]=10
# g[np.where(proj[4,:]==2)]=22
# g[np.where(proj[4,:]==3)]=33


'''

fill_rv = np.zeros([12,256,256])
for i in range(12):
     new_lv = fill_(img_lv[i,:])
     fill_rv[i,:] = new_lv
         
for i in range(2,3):
     plt.figure()
     plt.imshow(fill_rv[i,:])     


fill_myo = np.zeros([12,256,256])
for i in range(12):
     new_myo = fill_(img_myo[i,:])
     fill_myo[i,:] = new_myo


for i in range(2,3):
     plt.figure()
     plt.imshow(fill_myo[i,:]) 

for i in range(12):
     plt.figure()
     plt.imshow(img_myo[i,:])      

fill_rv = np.zeros([12,256,256])
for i in range(12):
     new_rv = fill_(img_rv[i,:])
     fill_rv[i,:] = new_rv


for i in range(3,4):
     plt.figure()
     plt.imshow(fill_rv[i,:]) 

for i in range(3,4):
     plt.figure()
     plt.imshow(img_rv[i,:]) 





temp = fill_rv[7,:].astype(np.uint8)
cnts = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnts[0][0]
(x1,y1),radius1 = cv2.minEnclosingCircle(cnt)

temp = img_lv[7,:].astype(np.uint8)
cnts = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnts[0][0]
(x,y),radius = cv2.minEnclosingCircle(cnt)


for j in range(4,5):
    print(j)
    T = fill_myo[j,:]
    if np.sum(T)!=0:
        T = T.astype(np.uint8)
        cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(cnts))
        radii = []
        for k in range(len(cnts)):
            cnt = cnts[0][k]
            (_,_),radius = cv2.minEnclosingCircle(cnt)
            print(radius)
            radii.append(radius)
            
max_radius = max(radii)   
radius_myo = int(radius)
LV_MYO = cv2.circle(LV, center, radius=radius_lv+radius_myo, color=2, thickness=2*radius_myo)


# for i in range(4,5):
#      plt.figure()
#      plt.imshow(LV_3d[i,:]) 


# for i in range(4,5):
#       plt.figure()
#       plt.imshow(img_rv[i,:])       


# for i in range(12):
#     if np.sum(img_rv[i,:])!=0:
#         a = draw_RV(img_rv[i,:])
#         LV_3d[i,:] = a'''

        
'''       
       
for i in range(4,5):
    plt.figure()
    plt.imshow(LV_3d[i,:])
    

    
g = LV_3d[4,:]
g[np.where(proj[4,:]==1)]=10
g[np.where(proj[4,:]==2)]=22
g[np.where(proj[4,:]==3)]=33





rv = np.zeros([256,256])
rv[np.where(proj[5,:]==3)]=1





RV = img_rv[7,:]

plt.figure()
plt.imshow(RV) 
    
    
RV = RV.astype(np.uint8)
cnts = cv2.findContours(RV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnts[0]
cv2.drawContours(RV,cnt,-1,(1,1,1),2)
kernel = np.ones((5,5),np.uint8)
T = cv2.morphologyEx(RV, cv2.MORPH_CLOSE, kernel)'''








