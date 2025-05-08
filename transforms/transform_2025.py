import SimpleITK as sitk
import nibabel as nib
import numpy as np

def Loadimage(imagename):
    nibimage = nib.load(imagename)
    imagedata = nibimage.get_fdata()
    numpyimage = np.array(imagedata).squeeze()

    return numpyimage

def extract_LA_from_SA(la="001_LA_ES.nii.gz",sa="001_SA_ES_gt.nii.gz",output_name="out.nii.gz"):

    LA_img = sitk.ReadImage(la)
    SA_img = sitk.ReadImage(sa)

    size = (LA_img.GetSize())
    new_img = sitk.Image(LA_img)

    # new_img = sitk.Image(size,LA_img.GetPixelID())
    # new_img.SetOrigin(LA_img.GetOrigin())
    # new_img.SetSpacing(LA_img.GetSpacing())

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            for z in range(0, size[2]):
                new_img[x, y, z] = 0
                point = LA_img.TransformIndexToPhysicalPoint([x, y, z])
                index_LA = SA_img.TransformPhysicalPointToIndex(point)
                if index_LA[0] < 0 or index_LA[0] >= SA_img.GetSize()[0]:
                    continue
                if index_LA[1] < 0 or index_LA[1] >= SA_img.GetSize()[1]:
                    continue
                if index_LA[2] < 0 or index_LA[2] >= SA_img.GetSize()[2]:
                    continue
                # print(index_LA)
                new_img[x, y, z] = SA_img[index_LA[0], index_LA[1], index_LA[2]]

    sitk.WriteImage(new_img, output_name)

def extract_SA_from_LA(la="001_LA_ED_gt.nii.gz",sa="001_SA_ED.nii.gz",output_name="out.nii.gz"):

    LA_img = sitk.ReadImage(la)
    SA_img = sitk.ReadImage(sa)

    size = (SA_img.GetSize())
    new_img = sitk.Image(SA_img)

    # new_img = sitk.Image(size,LA_img.GetPixelID())
    # new_img.SetOrigin(LA_img.GetOrigin())
    # new_img.SetSpacing(LA_img.GetSpacing())

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            for z in range(0, size[2]):
                new_img[x, y, z] = 0
                point = SA_img.TransformIndexToPhysicalPoint([x, y, z])
                index_LA = LA_img.TransformPhysicalPointToIndex(point)
                if index_LA[0] < 0 or index_LA[0] >= LA_img.GetSize()[0]:
                    continue
                if index_LA[1] < 0 or index_LA[1] >= LA_img.GetSize()[1]:
                    continue
                if index_LA[2] < 0 or index_LA[2] >= LA_img.GetSize()[2]:
                    continue
                # print(index_LA)
                new_img[x, y, z] = LA_img[index_LA[0], index_LA[1], index_LA[2]]

    sitk.WriteImage(new_img, output_name)

def reg_LA2SA(la="001_LA_ED_gt.nii.gz",sa="001_SA_ED.nii.gz",output_name="out.nii.gz"):

    LA_img = sitk.ReadImage(la)
    SA_img = sitk.ReadImage(sa)

    new_img = sitk.Image(LA_img)

    # size = (SA_img.GetSize())
    # new_img = sitk.Image(size, SA_img.GetPixelID())
    new_img.SetOrigin(SA_img.GetOrigin())
    new_img.SetSpacing(SA_img.GetSpacing())
    new_img.SetDirection(SA_img.GetDirection())

    sitk.WriteImage(new_img, output_name)

if __name__=="__main__":
    extract_LA_from_SA()

    LA_cine_numpyimg = Loadimage("LA_lab.nii.gz")
    SA_cine_numpyimg = Loadimage("001_SA_ED_gt.nii.gz")

    visualize_and_save = False
    if visualize_and_save == True:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(221)
        plt.imshow(LA_cine_numpyimg[:, :], cmap=plt.cm.gray)
        plt.subplot(222)
        plt.imshow(SA_cine_numpyimg[:, :, 3], cmap=plt.cm.gray)
        plt.subplot(223)
        plt.imshow(SA_cine_numpyimg[:, :, 4], cmap=plt.cm.gray)
        plt.subplot(224)
        plt.imshow(SA_cine_numpyimg[:, :, 5], cmap=plt.cm.gray)
        plt.show()
        #plt.savefig('img.png





