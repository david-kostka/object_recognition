# Zweck:

# Author: David Kostka
# Datum: 05.11.2020

import pandas as pd
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
import numpy as np
from time import time

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses

# Eigene Module
import visual as vis
import util
import model as md
import dataset as ds

# Definitionen für selbst generierte Daten (Modelle, Graphen, ...)
models_path = '../data/models/'
model_name = 'single_nao_5_best'
logs_path = '../data/logs/'
chkpts_path = '../data/checkpoints/'
graphs_path = '../data/graphs/'
tables_path = '../data/tables/'

# Datensatz Eigenschaften definieren
#ds.dataset_root_path = '/media/spl/WOLFPC-D/datasets/simulator/'
ds.dataset_root_path = 'E:/datasets/simulator/'
ds.csv_name = 'labels.csv'

ds.orig_size = (480, 640)
# Input muss % 4 == 0
ds.target_size = (60, 80)

# Namen der Unterordner des Simulator Datasets ('1_1/', '1_2/', ..., '2_5/')
subfolders = []
for i in range(1, 3):
    for j in range(1, 6):
        subfolders.append(str(i) + '_' + str(j) + '/')

for sub_folder in subfolders:
    dataset_path = ds.dataset_root_path + sub_folder
    ds.create_tfrecord_from_dir(dataset_path, include_negatives=True)
    #ds.create_tfrecord_from_dir(dataset_path)

def convert_to_uint8(img, label):
    return tf.cast(img * 255.0, dtype=tf.uint8), label

train_dset, val_dset = ds.load_combined_dataset(subfolders)
train_dset = train_dset.map(convert_to_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dset = val_dset.map(convert_to_uint8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dset_opt = train_dset.shuffle(buffer_size=512).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dset_opt = val_dset.shuffle(buffer_size=512).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def conf_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)
    bboxes_true = y_true[..., :4]
    bboxes_pred = y_pred[..., :4]
    conf_true = y_true[..., 4]
    conf_pred = y_pred[..., 4]
    #print(bboxes_true)
    #print(bboxes_true * K.expand_dims(conf_true))

    mseFunc = losses.MeanSquaredError(reduction='sum_over_batch_size')
    mse = mseFunc(bboxes_true * K.expand_dims(conf_true), bboxes_pred * K.expand_dims(conf_true))

    bceFunc = losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    bce = bceFunc(conf_true, conf_pred)
    #print("MSE: ", mse)
    #print("BCE: ", bce)
    return 10 * mse + bce

model = md.SingleNaoModel(ds.target_size, bnorm=False)

model.compile(loss=conf_loss,
              #optimizer=optimizer, 
              optimizer="adam", 
              #metrics=["accuracy"]
              )

callbacks = [   #ReduceLROnPlateau(verbose=1),
                EarlyStopping(patience=5, verbose=1, restore_best_weights=True, monitor='val_loss'),
                ModelCheckpoint(chkpts_path + model_name + '_{epoch}.tf', verbose=1, save_weights_only=True),
                TensorBoard(log_dir=logs_path + '{}'.format(time())) ]

model.fit(
    train_dset_opt,
    validation_data=val_dset_opt,
    epochs=50,
    callbacks=[callbacks]
)

# Save Model as File
model.save(models_path + model_name)

# Save as Format compatible with CompiledNN
model.save(models_path + model_name + '.hdf5', save_format='h5')

# Convert and Save to TFLite Model
converter = tf.lite.TFLiteConverter.from_saved_model(models_path + model_name)
tflite_model = converter.convert()
with open(models_path + model_name + '.tflite', 'wb') as f:
  f.write(tflite_model)