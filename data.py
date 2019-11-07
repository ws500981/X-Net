'''
data generator
ATLAS dataset has been transformed into .h5 format
'''

import numpy as np
import h5py
from matplotlib import pyplot as plt
#import time

def train_data_generator(patient_indexes, h5_file_path, batch_size):
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

    #start = time.process_time()
    while True:
        batch_img = []
        batch_label = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(slice_indexes)

            current_img = imgs[slice_indexes[i]][5:229, 2:194] #imgs and labels are 3d arrays
            current_label = labels[slice_indexes[i]][5:229, 2:194]
            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices
            #print('one batch done')
            #print('time taken to load one batch:',time.process_time()-start)

        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)
        #The yield statement suspends function’s execution and sends a value back to caller, but retains enough state to enable function to resume where it is left off. When resumed, the function continues execution immediately after the last yield run.


def create_train_date_generator(patient_indexes, h5_file_path, batch_size):
    return train_data_generator(patient_indexes, h5_file_path, batch_size)


def val_data_generator(patient_indexes, h5_file_path, batch_size=1):
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
            batch_img.append(current_img)
            batch_label.append(current_label)
            i = (i + 1) % num_of_slices
            #print('one batch done')
        yield np.expand_dims(np.array(batch_img), 3), np.expand_dims(np.array(batch_label), 3)


def create_val_date_generator(patient_indexes, h5_file_path, batch_size=1):
    return val_data_generator(patient_indexes, h5_file_path, batch_size)
