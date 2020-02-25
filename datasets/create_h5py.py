
import os
import cv2
import glob
import h5py
import string
#import PIL
import imageio
import argparse
import numpy as np
#from PIL import Image
import nibabel as nib



def save2h5py(ds_pth, h5py_pth):
    '''Convert raw ATLAS dataset into h5py file.'''

    # check input paths
    assert os.path.exists(ds_pth), 'The specified dataset path %s does NOT exists.'% ds_pth
    assert h5py_pth.endswith('.h5'), 'Invalid h5py file extention: %s. The file path shall end with \'.h5\''
    h5py_dir = os.path.dirname(h5py_pth)
    if not os.path.exists(h5py_dir):
        recursive_mkdir(h5py_dir)


    # Loop through each subject folder:
    # 1) add all label volumes tgr 
    # 2) save label to array  
    # 3) save lesion to array 
    ds_pth = ds_pth + '/c0004/c0004s0004t01'
    listofpaths = glob.glob(ds_pth, recursive=True) #get folder paths (each folder is for one brain with lesion+labels)
    assert len(listofpaths) > 0, 'No directory found that follows the defined pattern: %s. The glob results are: %s'%(ds_pth, str(listofpaths))
    listofpaths.sort()
    brainnumber = 0

    for i in range(len(listofpaths)):
        folder = listofpaths[i]
        print('progress: %d-th / %d: folder %s'%(i, len(listofpaths), folder), flush=True)
        
        lesionpath = folder+'/*t1w*.nii.gz'
        labelpath = folder+'/*LesionSmooth*.nii.gz'
        listoflesionpath = glob.glob(lesionpath,recursive = True) #get all lesion paths in this folder
        listoflabelpaths = glob.glob(labelpath,recursive = True) #get label path in this folder
        assert len(listoflesionpath) == 1, 'More than 1 or No brain image is found in folder %s: %s'%(folder, str(listoflesionpath))
        assert len(listoflesionpath) >= 1, 'No brain label is found in folder %s: %s'%(folder, str(listoflabelpath))

        # add all label images to one array
        labelimg_data = np.zeros((197,233,189)) #np.asarray([[[0]*189]*233]*197)
        for j in range(len(listoflabelpaths)):  
            list_a_item = listoflabelpaths[j]
            print(list_a_item)
            img = nib.load(list_a_item).get_fdata()
            print('label uniques: %s'%(str(np.unique(img, return_counts=True))))
            labelimg_data = np.add(labelimg_data, img)
            print('label_sum uniques: %s'%(str(np.unique(labelimg_data, return_counts=True))))
        print('arr.max() %.3f, %.3f'%(labelimg_data.max(),float(labelimg_data.max())) )
        print('arr.min() %.3f, %.3f'%(labelimg_data.min(),float(labelimg_data.min())) )
        labelimg_data = np.round(labelimg_data).astype(np.uint8) #set as unsigned int8 first before normalization because after normalization everything will be less than 1 so they will become 0 with astype(np.uint8)
        print('arr.max() %.3f, %.3f'%(labelimg_data.max(),float(labelimg_data.max())) )
        print('arr.min() %.3f, %.3f'%(labelimg_data.min(),float(labelimg_data.min())) )
        labelimg_data = normalise(labelimg_data, new_max=1, new_min=0) #normalize the data from range [0,255] to range[0,1]
        print('label_sum_norm uniques: %s'%(str(np.unique(labelimg_data,return_counts=True))))
        labelimg_data = np.swapaxes(labelimg_data,0,2) #swap the axes such that the first dimension is the slice number
        if brainnumber == 0: #save all slices (239 brains)*(189 for each brain) into a single array
            label_array = labelimg_data
            print('label',np.shape(label_array))
        elif brainnumber != 0:
            label_array = np.concatenate((label_array, labelimg_data))
            print('label',np.shape(label_array))


        # save all label images to one array
        print(listoflesionpath[0])
        lesionimg = nib.load(listoflesionpath[0])
        lesionimg_data = lesionimg.get_fdata()
        print('arr.max() %.3f, %.3f'%(lesionimg_data.max(),float(lesionimg_data.max())) )
        print('arr.min() %.3f, %.3f'%(lesionimg_data.min(),float(lesionimg_data.min())) )
        #lesionimg_data = lesionimg_data.astype(np.uint8)
        print('arr.max() %.3f, %.3f'%(lesionimg_data.max(),float(lesionimg_data.max())) )
        print('arr.min() %.3f, %.3f'%(lesionimg_data.min(),float(lesionimg_data.min())) )
        lesionimg_data = normalise(lesionimg_data, new_max=1, new_min=-1)
        print('arr.max() %.3f, %.3f'%(lesionimg_data.max(),float(lesionimg_data.max())) )
        print('arr.min() %.3f, %.3f'%(lesionimg_data.min(),float(lesionimg_data.min())) )
        lesionimg_data = np.swapaxes(lesionimg_data,0,2) #swap the axes such that the first dimension is the slice number
        if brainnumber == 0: #save all slices (239 brains)*(189 for each brain) into a single array
            lesion_array = lesionimg_data
            print('lesion',np.shape(lesion_array), '\n')
        elif brainnumber != 0:
            lesion_array = np.concatenate((lesion_array, lesionimg_data))
            print('lesion',np.shape(lesion_array),'\n')

        brainnumber += 1

    # write arrays to h5py
    hdf5_file = h5py.File(h5py_pth, 'w')

    lesion_shape = (lesion_array.shape[0], 233, 197)
    hdf5_file.create_dataset("lesion", lesion_shape, data = lesion_array)

    label_shape = (label_array.shape[0], 233, 197)
    hdf5_file.create_dataset("label", label_shape, data = label_array)

    hdf5_file.close()
    print('End converting to h5py. There are %d subjects in total.'%brainnumber)




def check_h5py(h5py_pth, save_pth):
    '''Visualize h5py file through extracting arrays into .png files'''


    assert os.path.exists(h5py_pth), 'The h5py fath does NOT exists.: %s'%h5py_pth
    assert h5py_pth.endswith('.h5'), 'Invalid h5py file extention: %s. The file path shall end with \'.h5\''
    if not os.path.exists(save_pth):
        recursive_mkdir(save_pth)

    hdf5_file = h5py.File(h5py_pth, 'r')
    Lesion = hdf5_file['lesion']
    Label = hdf5_file['label']

    lesion = np.array(Lesion)
    label = np.array(Label)

    assert lesion.shape == label.shape
    print('3d_shape',np.shape(lesion))
    print('3d_shape',np.shape(label))
    whatevernumber = 0
 
    for whatevernumber in range(lesion.shape[0]):
        current_img = lesion[whatevernumber]
        current_label = label[whatevernumber]

        print('current_img.max() %.3f, %.3f'%(current_img.max(),float(current_img.max())) )
        print('current_img.min() %.3f, %.3f'%(current_img.min(),float(current_img.min())) )
        print('current_label.max() %.3f, %.3f'%(current_label.max(),float(current_label.max())) )
        print('current_label.min() %.3f, %.3f'%(current_label.min(),float(current_label.min())) )
        current_img = normalise(current_img, new_max=255, new_min=0) 
        current_label =  current_label * 255
        current_img = current_img.astype(np.uint8)
        current_label = current_label.astype(np.uint8)

        assert current_img.shape == current_label.shape
        save_loaded_img_path =  str(whatevernumber) + 'img_' + '.png'
        save_loaded_label_path = str(whatevernumber) + 'label_' + '.png'
        full_img_path = os.path.join(save_pth, save_loaded_img_path)
        full_label_path = os.path.join(save_pth, save_loaded_label_path)
        imageio.imwrite(full_img_path, current_img)
        imageio.imwrite(full_label_path, current_label)
        whatevernumber += 1
    hdf5_file.close()
    print('Images are saved to %s'%save_pth)




def recursive_mkdir(dir):
    '''Recursively mkdir. E.g. dir=/a/b/c/d, if only /a/b exists, /a/b/c and /a/b/c/d/ will be created.'''

    if os.path.exists(dir):
        return 

    path_splitted = os.path.split(os.path.abspath(dir))
    print('recursive_mkdir',path_splitted)

    if not os.path.exists(path_splitted[0]):
        recursive_mkdir(path_splitted[0])
    
    os.mkdir(dir)
    print('Create dir: %s'%dir)



def normalise(array, new_min, new_max):
    """Rescale an array in the min and max value defined

    params:
        array:       array to process
        new_min:   new minimum
        new_max:   new maximum

    return:            rescaled array
    """
    old_max = float(array.max())
    old_min = float(array.min())

    if old_max == old_min:
        return np.zeros_like(array) + new_min
    
    array = (new_max - new_min) * (array - old_min) / (old_max - old_min) + new_min

    assert abs(array.max() - new_max) < 0.001 , 'Normalization fails: requested new_max=%.3f, resulted new_max=%.3f, old_max=%.3f'%(new_max, array.max(), old_max)
    assert abs(array.min() - new_min) < 0.001 , 'Normalization fails: requested new_min=%.3f, resulted new_min=%.3f, old_min=%.3f'%(new_min, array.min(), old_min)

    return array


def main(args):
    ds_pth = args.ds_pth
    h5py_pth = args.h5py_pth

    save2h5py(ds_pth, h5py_pth)

    if args.run_check:
        check_h5py(h5py_pth = h5py_pth, save_pth = args.check_save_pth)  


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='X-Net.')
    parser.add_argument('--ds_pth',default='/home/wwu009/Project/Atlas1.2standard_MNI(2.0)',help='Path of raw dataset')
    parser.add_argument('--h5py_pth', default='/home/wwu009/Project/hd5/normalized_file.h5', help='The full path of converted h5py data file')
    parser.add_argument('--run_check', action='store_true',help='Whether to run checking function')
    parser.add_argument('--check_save_pth', default='/home/wwu009/Project/Dataloader/save_loaded', help='Required only when run_check is true. The path to save images for checking')


    args = parser.parse_args()
    main(args)
