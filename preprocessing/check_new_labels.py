for i in range(40):
    a1 = next(a)
    
    org = a1[0][0,:].numpy()
    cropped = a1[1][0,:].numpy()
    org_shape = a1[2]
    
    DIM_ = 256
    def Back_to_orgSize(img,org_shape):
        if org_shape[1]>DIM_ and org_shape[2]>DIM_:
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            d2 = int((org_shape[2]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            n2 = int(org_shape[2]-d2)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,d1:n1,d2:n2] = img
            
          
        if org_shape[1]<DIM_ and org_shape[2]<DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)
            d2 = int((DIM_-org_shape[2])/2)
            n1 = int(DIM_-d1)
            n2 = int(DIM_-d2)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp = img[:,d1:n1,d2:n2]
            
        if org_shape[1]>DIM_ and org_shape[2]==DIM_:
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
            temp[:,d1:n1,:] = img
            
        if org_shape[1]==DIM_ and org_shape[2]==DIM_:
            temp = img
            
        if org_shape[1]==DIM_ and org_shape[2]<DIM_:
            temp = np.zeros(org_shape)
            d2 = int((DIM_-org_shape[2])/2)
            n2 = int(DIM_-d2)
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp = img[:,:,d2:n2]
            

        if org_shape[1]>DIM_ and org_shape[2]<DIM_:
            
            temp = np.zeros(org_shape)
            d1 = int((org_shape[1]-DIM_)/2)
            n1 = int(org_shape[1]-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
                
                ## for smaller side 
            d2 = int((DIM_-org_shape[2])/2)
            n2 = int(DIM_-d2)
            
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,d1:n1,:] = img[:,:,d2:n2]
            
        if org_shape[1]<DIM_ and org_shape[2]>DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)
            n1 = int(DIM_-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
           
            d2 = int((org_shape[2]-DIM_)/2)
            n2 = int(org_shape[2]-d2)
            if( org_shape[2]%2)!=0:
                n2=n2-1
            temp[:,:,d2:n2] = img[:,d1:n1,:]

        if org_shape[1]<DIM_ and org_shape[2]==DIM_:
            temp = np.zeros(org_shape)
            d1 = int((DIM_-org_shape[1])/2)     
            n1 = int(DIM_-d1)
            if( org_shape[1]%2)!=0:
                n1=n1-1
    
            temp = img[:,d1:n1,:]
            
        return temp
    
    back = Back_to_orgSize(cropped,org_shape)
    
    # plt.figure()
    # plt.imshow(back[0,:])
    
    # plt.figure()
    # plt.imshow(org[0,:])
    
    s1 = np.array_equal(back, org)
    print(int(s1))
    
    back_label = generate_label_3(back, 17)
    org_label = generate_label_3(org, 17)

    s1 = np.array_equal(back_label,org_label)
    print(int(s1))
    print("   ")
