'''
data generator
ATLAS dataset has been transformed into .h5 format
'''

import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
import imageio
#import time

def train_data_generator(patient_indexes, h5_file_path, batch_size, ck_path):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['lesion']
    labels = file['label']

    # 输入的是病人的index，转换成切片的index
    slice_indexes = []
    for patient_index in patient_indexes: #229 patients, patient_indexes is an array [0,1,2,...228]
        for slice_index in range(189): #each patient has 189 slices (0,1,2,...188)
            slice_indexes.append(patient_index * 189 + slice_index) #put each slice number into slice_indexes, all together 229*189 slices
    num_of_slices = len(slice_indexes) #229*189 slices in total
    print(num_of_slices)
    # the above comment need to be amemded because patient_indexes is not the total no of patients, only those for train but not val

    #start = time.process_time()
    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(slice_indexes)

            current_img = imgs[slice_indexes[i]][5:229, 2:194] #imgs and labels are 3d arrays
            current_label = labels[slice_indexes[i]][5:229, 2:194]

            save_loaded_path = ck_path + '/saved_loaded_train/'
            if not os.path.exists(save_loaded_path):
                os.mkdir(save_loaded_path)
            #print('loaded_train_yes',current_img.shape)
            #print('loaded_train_yes',current_label.shape) #they should be 2d arrays
            twod_img = current_img * 255
            twod_label = current_label * 255
            twod_img = twod_img.astype(np.uint8)
            twod_label = twod_label.astype(np.uint8)
            assert len(twod_img) == len(twod_label)
            save_loaded_img_path =  str(i) + 'img_' + '.png'
            save_loaded_label_path = str(i) + 'label_' + '.png'       
            full_img_path = os.path.join(save_loaded_path,save_loaded_img_path)
            full_label_path = os.path.join(save_loaded_path,save_loaded_label_path)
            imageio.imwrite(full_img_path, twod_img)
            imageio.imwrite(full_label_path, twod_label)

            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices
            #print('one batch done')
            #print('time taken to load one batch:',time.process_time()-start)

        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)
        #The yield statement suspends function’s execution and sends a value back to caller, but retains enough state to enable function to resume where it is left off. When resumed, the function continues execution immediately after the last yield run.


def create_train_date_generator(patient_indexes, h5_file_path, batch_size, ck_path):
    return train_data_generator(patient_indexes, h5_file_path, batch_size, ck_path)


def val_data_generator(patient_indexes, h5_file_path, ck_path, batch_size=1):
    i = 0
    file = h5py.File(h5_file_path, 'r')
    imgs = file['lesion']
    labels = file['label']

    # 输入的是病人的index，转换成切片的index
    slice_indexes = []
    for patient_index in patient_indexes:
        for slice_index in range(189):
            slice_indexes.append(patient_index * 189 + slice_index)
    num_of_slices = len(slice_indexes)

    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            current_img = imgs[slice_indexes[i]][5:229, 2:194]
            current_label = labels[slice_indexes[i]][5:229, 2:194]

            save_loaded_path = ck_path + '/saved_loaded_val/'
            if not os.path.exists(save_loaded_path):
                os.mkdir(save_loaded_path)
            print('loaded_val_yes',current_img.shape)
            print('loaded_val_yes',current_label.shape)
            twod_img = current_img * 255
            twod_label = current_label * 255
            twod_img = twod_img.astype(np.uint8)
            twod_label = twod_label.astype(np.uint8)
            assert len(twod_img) == len(twod_label)
            save_loaded_img_path =  str(i) + 'img_' + '.png'
            save_loaded_label_path = str(i) + 'label_' + '.png'       
            full_img_path = os.path.join(save_loaded_path,save_loaded_img_path)
            full_label_path = os.path.join(save_loaded_path,save_loaded_label_path)
            imageio.imwrite(full_img_path, twod_img)
            imageio.imwrite(full_label_path, twod_label)

            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices
            #print('one batch done')
        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)


def create_val_date_generator(patient_indexes, h5_file_path, ck_path, batch_size=1):
    return val_data_generator(patient_indexes, h5_file_path, ck_path, batch_size)
