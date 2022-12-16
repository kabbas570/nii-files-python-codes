import scipy
import scipy.ndimage
import nibabel as nib
import os

new_dir_path = r'C:\My_Data\M2M Data\data\data_2\test'
    
# for i in range(99,100):
    
#     #path = r'C:\My_Data\M2M Data\data\train/'+str(i)+'/'+str(i)
#     path1 = r'C:\My_Data\M2M Data\data\train/'+'0'+str(i)+'/0'+str(i)
#     path = path1 + '_SA_ES.nii.gz'
#     SA_ES = nib.load(path)
#     qform = SA_ES.get_qform()
#     SA_ES.set_qform(qform)
#     sform  = SA_ES.get_sform()
#     SA_ES.set_sform(sform)
    
#     path = path1 +'_SA_ES_gt.nii.gz'
#     SA_ES_gt = nib.load(path)
#     qform = SA_ES_gt.get_qform()
#     SA_ES_gt.set_qform(qform)
#     sform  = SA_ES_gt.get_sform()
#     SA_ES_gt.set_sform(sform)
    
#     path = path1 +'_LA_ES.nii.gz'
#     LA_ES = nib.load(path)
#     qform = LA_ES.get_qform()
#     LA_ES.set_qform(qform)
#     sform  = LA_ES.get_sform()
#     LA_ES.set_sform(sform)
    
#     path = path1 +'_LA_ES_gt.nii.gz'
#     LA_ES_gt = nib.load(path)
#     qform = LA_ES_gt.get_qform()
#     LA_ES_gt.set_qform(qform)
#     sform  = LA_ES_gt.get_sform()
#     LA_ES_gt.set_sform(sform)
    
#       ### for ED ###
#     path = path1 +'_SA_ED.nii.gz'
#     SA_ED = nib.load(path)
#     qform = SA_ED.get_qform()
#     SA_ED.set_qform(qform)
#     sform  = SA_ED.get_sform()
#     SA_ED.set_sform(sform)
    
#     path = path1 +'_SA_ED_gt.nii.gz'
#     SA_ED_gt = nib.load(path)
#     qform = SA_ED_gt.get_qform()
#     SA_ED_gt.set_qform(qform)
#     sform  = SA_ED_gt.get_sform()
#     SA_ED_gt.set_sform(sform)
    
#     path = path1 +'_LA_ED.nii.gz'
#     LA_ED = nib.load(path)
#     qform = LA_ED.get_qform()
#     LA_ED.set_qform(qform)
#     sform  = LA_ED.get_sform()
#     LA_ED.set_sform(sform)
    
#     path = path1 +'_LA_ED_gt.nii.gz'
#     LA_ED_gt = nib.load(path)
#     qform = LA_ED_gt.get_qform()
#     LA_ED_gt.set_qform(qform)
#     sform  = LA_ED_gt.get_sform()
#     LA_ED_gt.set_sform(sform)
    

#     os.mkdir(new_dir_path+'/'+'0'+str(i))
#     path = new_dir_path + '/'+'0'+str(i)
    

#     nib.save(SA_ES,os.path.join(path+'/'+'0'+str(i)+"_SA_ES.nii.gz"))
#     nib.save(SA_ES_gt,os.path.join(path+'/'+'0'+str(i)+"_SA_ES_gt.nii.gz"))
    
#     nib.save(LA_ES,os.path.join(path+'/'+'0'+str(i)+"_LA_ES.nii.gz"))
#     nib.save(LA_ES_gt,os.path.join(path+'/'+'0'+str(i)+"_LA_ES_gt.nii.gz"))
    
    
#     nib.save(SA_ED,os.path.join(path+'/'+'0'+str(i)+"_SA_ED.nii.gz"))
#     nib.save(SA_ED_gt,os.path.join(path+'/'+'0'+str(i)+"_SA_ED_gt.nii.gz"))
    
#     nib.save(LA_ED,os.path.join(path+'/'+'0'+str(i)+"_LA_ED.nii.gz"))
#     nib.save(LA_ED_gt,os.path.join(path+'/'+'0'+str(i)+"_LA_ED_gt.nii.gz"))




for i in range(201,361):
    path1 = r'C:\My_Data\M2M Data\data\test/'+str(i)+'/'+str(i)
    path = path1 + '_SA_ES.nii.gz'
    SA_ES = nib.load(path)
    qform = SA_ES.get_qform()
    SA_ES.set_qform(qform)
    sform  = SA_ES.get_sform()
    SA_ES.set_sform(sform)
    
    path = path1 +'_SA_ES_gt.nii.gz'
    SA_ES_gt = nib.load(path)
    qform = SA_ES_gt.get_qform()
    SA_ES_gt.set_qform(qform)
    sform  = SA_ES_gt.get_sform()
    SA_ES_gt.set_sform(sform)
    
    path = path1 +'_LA_ES.nii.gz'
    LA_ES = nib.load(path)
    qform = LA_ES.get_qform()
    LA_ES.set_qform(qform)
    sform  = LA_ES.get_sform()
    LA_ES.set_sform(sform)
    
    path = path1 +'_LA_ES_gt.nii.gz'
    LA_ES_gt = nib.load(path)
    qform = LA_ES_gt.get_qform()
    LA_ES_gt.set_qform(qform)
    sform  = LA_ES_gt.get_sform()
    LA_ES_gt.set_sform(sform)
    
      ### for ED ###
    path = path1 +'_SA_ED.nii.gz'
    SA_ED = nib.load(path)
    qform = SA_ED.get_qform()
    SA_ED.set_qform(qform)
    sform  = SA_ED.get_sform()
    SA_ED.set_sform(sform)
    
    path = path1 +'_SA_ED_gt.nii.gz'
    SA_ED_gt = nib.load(path)
    qform = SA_ED_gt.get_qform()
    SA_ED_gt.set_qform(qform)
    sform  = SA_ED_gt.get_sform()
    SA_ED_gt.set_sform(sform)
    
    path = path1 +'_LA_ED.nii.gz'
    LA_ED = nib.load(path)
    qform = LA_ED.get_qform()
    LA_ED.set_qform(qform)
    sform  = LA_ED.get_sform()
    LA_ED.set_sform(sform)
    
    path = path1 +'_LA_ED_gt.nii.gz'
    LA_ED_gt = nib.load(path)
    qform = LA_ED_gt.get_qform()
    LA_ED_gt.set_qform(qform)
    sform  = LA_ED_gt.get_sform()
    LA_ED_gt.set_sform(sform)
    

    os.mkdir(new_dir_path+'/'+str(i))
    path = new_dir_path + '/'+str(i)
    

    nib.save(SA_ES,os.path.join(path+'/'+str(i)+"_SA_ES.nii.gz"))
    nib.save(SA_ES_gt,os.path.join(path+'/'+str(i)+"_SA_ES_gt.nii.gz"))
    
    nib.save(LA_ES,os.path.join(path+'/'+str(i)+"_LA_ES.nii.gz"))
    nib.save(LA_ES_gt,os.path.join(path+'/'+str(i)+"_LA_ES_gt.nii.gz"))
    
    
    nib.save(SA_ED,os.path.join(path+'/'+str(i)+"_SA_ED.nii.gz"))
    nib.save(SA_ED_gt,os.path.join(path+'/'+str(i)+"_SA_ED_gt.nii.gz"))
    
    nib.save(LA_ED,os.path.join(path+'/'+str(i)+"_LA_ED.nii.gz"))
    nib.save(LA_ED_gt,os.path.join(path+'/'+str(i)+"_LA_ED_gt.nii.gz"))
