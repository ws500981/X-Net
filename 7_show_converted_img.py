import nibabel as nib
import os
import glob
import numpy as np
import cv2
import string
import PIL
from PIL import Image
import imageio
import h5py


all_paths = glob.glob('/home/wwu009/Project/Atlas1.2standard_MNI(2.0)/*/c*/*', recursive=True) #get folder paths (each folder is for one brain with lesions+label)
all_paths.sort()

#new (all lesion slices tgt) & (save lesion to array) & (save label to array) in the folder
brainnumber = 0
listofpaths = all_paths #folder paths, each folder contains lesions and one label

for i in range(len(listofpaths)):
    folder = listofpaths[i]
    lesionpath = folder+'/*LesionSmooth*.nii.gz'
    labelpath = folder+'/*t1w*.nii.gz'
    listoflesionpaths = glob.glob(lesionpath,recursive = True) #get all lesion paths in this folder
    listoflabelpath = glob.glob(labelpath,recursive = True) #get label path in this folder

    #save all lesion images to one array
    lesionimg_data = np.asarray([[[0]*189]*233]*197)
    for j in range(len(listoflesionpaths)): #for each lesion, extract all lesion data and add them all tgt
        list_a_item = listoflesionpaths[j]
        print(list_a_item)
        img = nib.load(list_a_item)
        lesionimg_data = np.add(lesionimg_data, img.get_fdata())
    lesionimg_data = lesionimg_data.astype(np.uint8) #set as unsigned int8 first before normalization because after normalization everything will be less than 1 so they will become 0 with astype(np.uint8)
    lesionimg_data = (lesionimg_data-lesionimg_data.min())/(lesionimg_data.max()-lesionimg_data.min()) #normalize the data from range [0,255] to range[0,1]
    lesionimg_data = np.swapaxes(lesionimg_data,0,2) #swap the axes such that the first dimension is the slice number
    if brainnumber == 0: #save all slices (239 brains)*(189 for each brain) into a single array
        lesion_array = lesionimg_data
        print('lesion',np.shape(lesion_array))
    elif brainnumber != 0:
        lesion_array = np.concatenate((lesion_array, lesionimg_data))
        print('lesion',np.shape(lesion_array))



    #save all label images to one array
    print(listoflabelpath[0])
    labelimg = nib.load(listoflabelpath[0])
    labelimg_data = labelimg.get_fdata()
    labelimg_data = labelimg_data.astype(np.uint8)
    labelimg_data = (labelimg_data-labelimg_data.min())/(labelimg_data.max()-labelimg_data.min())
    labelimg_data = np.swapaxes(labelimg_data,0,2) #swap the axes such that the first dimension is the slice number
    if brainnumber == 0: #save all slices (239 brains)*(189 for each brain) into a single array
        label_array = labelimg_data
        print('label',np.shape(label_array), '\n')
    elif brainnumber != 0:
        label_array = np.concatenate((label_array, labelimg_data))
        print('label',np.shape(label_array),'\n')


    brainnumber += 1



folder = 'hd5'
path = os.path.join('/home/wwu009/Project',folder)
if not os.path.exists(path):
    os.mkdir(path)
path = os.path.join(path,'normalized_file.h5')
#print(path)
hdf5_file = h5py.File(path, 'w')


lesion_shape = (lesion_array.shape[0], 233, 197)
hdf5_file.create_dataset("lesion", lesion_shape, data = lesion_array)

label_shape = (label_array.shape[0], 233, 197)
hdf5_file.create_dataset("label", label_shape, data = label_array)

hdf5_file.close()


hdf5_file = h5py.File('/home/wwu009/Project/hd5/normalized_file.h5', 'r')
Lesion = hdf5_file['lesion']
Label = hdf5_file['label']

lesion = np.array(Lesion)
label = np.array(Label)

assert lesion.shape == label.shape
print('3d_shape',np.shape(lesion))
print('3d_shape',np.shape(label))
whatevernumber = 0
save_loaded_path = '/home/wwu009/Project/Dataloader/save_loaded'
if not os.path.exists(save_loaded_path):
    os.mkdir(save_loaded_path)
for whatevernumber in range(384):
    current_img = label[whatevernumber]
    current_label = lesion[whatevernumber]
    #print('2d_shape',current_img.shape)
    #print('2d_shape',current_label.shape)
    current_img = current_img*255
    current_label = current_label*255
    current_img = current_img.astype(np.uint8)
    current_label = current_label.astype(np.uint8)
    assert current_img.shape == current_label.shape
    save_loaded_img_path =  str(whatevernumber) + 'img_' + '.png'
    save_loaded_label_path = str(whatevernumber) + 'label_' + '.png'
    full_img_path = os.path.join(save_loaded_path,save_loaded_img_path)
    full_label_path = os.path.join(save_loaded_path,save_loaded_label_path)
    imageio.imwrite(full_img_path, current_img)
    imageio.imwrite(full_label_path, current_label)
    whatevernumber += 1
hdf5_file.close()
