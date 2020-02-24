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


hdf5_file = h5py.File('/home/wwu009/Project/hd5/normalized_file2.0.h5', 'r')
Lesion = hdf5_file['lesion']
Label = hdf5_file['label']

lesion = np.array(Lesion)
label = np.array(Label)
assert not np.any(np.isnan(lesion)),'lesion got nan!'
assert not np.any(np.isnan(label)), 'label got nan!'
assert lesion.shape == label.shape
print('3d_shape',np.shape(lesion))
print('3d_shape',np.shape(label))
whatevernumber = 0
save_loaded_path = '/home/wwu009/Project/Dataloader_forxnet/save_loaded'
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
