import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard, LambdaCallback
import os
import pandas as pd
from sklearn.model_selection import KFold

from utils import get_score_from_all_slices
from model import create_xception_unet_n
from loss import get_loss, dice
from data import create_train_date_generator, create_val_date_generator

import argparse

data_file_path = '/home/wwu009/Project/hd5/normalized_file.h5'
pretrained_weights_file = None
input_shape = (224, 192, 1)
batch_size = 8
num_folds = 5


def train(ck_dir, fold, train_patient_indexes, val_patient_indexes):

    log_dir = os.path.join(ck_dir,'fold_' + str(fold) + '/') #train(...) will be called for each fold, that is for folds 0,1,2,3,4
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    num_slices_train = len(train_patient_indexes) * 189
    num_slices_val = len(val_patient_indexes) * 189

    # Create model
    K.clear_session()
    model = create_xception_unet_n(input_shape=input_shape, pretrained_weights_file=pretrained_weights_file)
    model.compile(optimizer=Adam(lr=1e-3), loss=get_loss, metrics=[dice]) #Before training a model, you need to configure the learning process, which is done via the compile method

    # Get callbacks
    checkpoint = ModelCheckpoint(log_dir + 'ep={epoch:03d}-loss={loss:.3f}-val_loss={val_loss:.3f}.h5', verbose=1,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=1e-3, patience=3, verbose=1)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    csv_logger = CSVLogger(log_dir + 'record.csv')
    tensorboard = TensorBoard(log_dir=log_dir)

    # train the model
    model.fit_generator(
        create_train_date_generator(patient_indexes=train_patient_indexes, h5_file_path=data_file_path, batch_size=batch_size),
        steps_per_epoch=max(1, num_slices_train // batch_size),
        validation_data=create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path, batch_size=9),
        validation_steps=max(1, num_slices_val // 9),
        epochs=100,
        initial_epoch=0,
        callbacks=[checkpoint, reduce_lr, tensorboard, csv_logger]) #early_stopping, tensorboard, csv_logger])
    model.save_weights(log_dir + 'trained_final_weights.h5')

    # Evaluate model
    predicts = []
    labels = []
    f = create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    for _ in range(num_slices_val):
        img, label = f.__next__()
        predicts.append(model.predict(img))
        labels.append(label)
    predicts = np.array(predicts)
    labels = np.array(labels)
    score_record = get_score_from_all_slices(labels=labels, predicts=predicts)

    # save score
    df = pd.DataFrame(score_record)
    df.to_csv(os.path.join(log_dir, 'score_record.csv'), index=False)

    # print score
    mean_score = {}
    for key in score_record.keys():
        print('In fold ', fold, ', average', key, ' value is: \t ', np.mean(score_record[key]))
        mean_score[key] = np.mean(score_record[key])

    # exit training
    K.clear_session()
    return mean_score


def main(args):
    # create checkpoint
    ck_path = './checkpoints/'+args.exp_nm
    if not os.path.exists(ck_path):
        os.mkdir(ck_path)

    # prepare indexes of patients for training and validation, respectively
    num_patients = 239
    patients_indexes = np.array([i for i in range(num_patients)]) #patients_indexes is a 1d array containing 239 numbers (0-238)
    kf = KFold(n_splits=num_folds, shuffle=False) #num_folds = 5
    #K-Folds cross-validator. Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default). Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    # train, and record the scores of each fold
    folds_score = []
    for fold, (train_patient_indexes, val_patient_indexes) in enumerate(kf.split(patients_indexes)):
        #kf.split(patients_indexes) is a 2d array-like thing, if num_folds = 5, kf.split(...) will contain 5 folds, each fold contains a pair of ndarray of train and validation indices
        fold_mean_score = train(ck_dir =ck_path, fold=fold, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes) #for each fold of the 5, train & validate the model and return mean score, mean score is a dictionary
        folds_score.append(fold_mean_score) #put mean score for each of the 5 folds in one list

    # calculate average score
    print('Final score from ', num_folds, ' folds cross validation:')
    final_score = {} #create an empty dictionary
    for key in folds_score[0].keys():
        scores = []
        for i in range(num_folds):
            scores.append(folds_score[i][key])
        final_score[key] = np.mean(scores)
        print(key, ' score: \t', final_score[key])

    # save final score
    df = pd.DataFrame(final_score, index=[0])
    df.to_csv(ck_path,'x_net_final_score.csv', index=False)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser(description='X-Net.')
    parser.add_argument('--exp_nm',required=True ,help='experiment name')


    args = parser.parse_args()
    main(args)
