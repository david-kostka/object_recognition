# Zweck:
# Hilfsfunktionen zum erstellen von TFRecords aus Simulator Trainingsdaten
# Erzeugung eines tf.data.Dataset Objektes basierend auf den TFRecord Files und Bilddaten
# Author: David Kostka
# Datum: 05.11.2020

import pandas as pd
import tensorflow as tf
import numpy as np
import os
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt

dataset_root_path = 'E:/datasets/simulator/'
csv_name = 'labels.csv'

orig_size = (480, 640)
target_size = (60, 80)

#class SingleNaoDataset:
def create_tf_example(row):
    '''
    Konvertiert ein DataFrame Row in ein TFRecord Example
    '''
    filename = row.name.encode('utf8')
    xmins = row.minX / orig_size[1]
    xmaxs = row.maxX / orig_size[1]
    ymins = row.minY / orig_size[0]
    ymaxs = row.maxY / orig_size[0]

    classes = row.classes

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[xmins])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[xmaxs])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[ymins])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[ymaxs])),
        'image/object/class/label': tf.train.Feature(float_list=tf.train.FloatList(value=[classes]))
    }))
    return tf_example

def write_tfrecord(labels, name, path):
    '''
    Schreibt 'labels' als Examples in eine TFRecord File
    '''
    writer = tf.io.TFRecordWriter(path + name)
    for label in labels.itertuples():
        tf_example = create_tf_example(label)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print("TFRecord '" + path + name + "' created")

def parse_image(name, filepath, rgb=False, resize=True):
    '''
    Ladet und Decodiert ein Bild aus dem 'filepath'
    Mit rgb kann angegeben werden ob das Bild 1 (grayscale) oder 3 (RGB) Kanäle haben soll
    '''
    image = tf.io.read_file(filepath + name)
    image = tf.image.decode_png(image)
    if not rgb: image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if resize: image = tf.image.resize(image, target_size)
    return image

def parse_tfrecord(tfrecord, filepath):
    '''
    Lese aus dem 'tfrecord' das Bild und Labels aus
    '''
    # Aufbau eines TFRecord Example
    IMAGE_FEATURE_MAP = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.float32)
    }

    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

    x_train = parse_image(x['image/filename'], filepath + 'images/')

    y_train = [ x['image/object/bbox/xmin'],
                x['image/object/bbox/ymin'],
                x['image/object/bbox/xmax'],
                x['image/object/bbox/ymax'],
                x['image/object/class/label']
                ]
    
    #print(y_train)

    #y_train = np.asarray(y_train)
    #print(x_train)
    #print(y_train)

    return x_train, y_train

def load_tfrecord_dataset(filepath, csv_name):
    '''
    Erzeugt ein tf.data.Dataset aus einer TFRecord File und ersetzt dabei den Bildnamen mit den tatsächlichen Bilddaten
    Die Bilder müssen in einem Unterordner 'images/' relativ zum 'filepath' vorhanden sein.
    '''
    dataset = tf.data.TFRecordDataset(filepath + csv_name)
    #print('Loaded Dataset: ' + filepath)
    return dataset.map(lambda x: parse_tfrecord(x, filepath), num_parallel_calls=tf.data.experimental.AUTOTUNE)

def create_tfrecord_from_dir(path, include_negatives=False):
    '''
    Erzeugt Train und Validation TFRecord Files aus Label Datei in 'path'
    Dabei werden die Einträge so gefiltert das nur Bildname/Label Einträge übrig bleiben, die nur einen einzigen Nao im Bild haben
    
    Input: Die Label Datei muss eine CSV Datei mit Spalten der Form ('name', 'minX', 'minY', 'maxX', 'maxY') beinhalten.
    Dabei ist 'name' der Bilddateipfad und der Rest die BBOX Koordinaten

    Output: TFRecord Files 'train.record' und 'val.record' mit Einträgen der Form (Bildname, BBOX Koordinaten), für Bilder, bei denen nur ein Nao zu sehen ist.
    '''
    raw_data = pd.read_csv(path + csv_name, sep=r'\s*,\s*', index_col=None)
    raw_data = raw_data[['name', 'minX', 'minY', 'maxX', 'maxY']]
    
    if include_negatives:
        raw_data['classes'] = 1
        labels_img_names = raw_data['name'].unique()
        all_img_names = os.listdir(path + 'images/')
        no_nao_img_names = np.setdiff1d(all_img_names,labels_img_names)
        new_data = pd.DataFrame(no_nao_img_names, columns=['name'])
        new_data['classes'] = 0
        raw_data = raw_data.append(new_data).sample(frac=1).reset_index(drop=True)
        raw_data = raw_data[raw_data.name != 'name']

    data = raw_data.fillna(0).groupby('name').filter(lambda x: len(x) == 1)
    data = data.sample(frac=1).reset_index(drop=True)

    count = len(data)
    split_index = int(count - count * 0.2)

    print(data)

    write_tfrecord(data[:split_index], 'train.record', path)
    write_tfrecord(data[split_index:], 'val.record', path)

def load_combined_dataset(subfolders):
    '''
    Kombiniert alle TFRecord Files der Ordner in 'subfolder' zu einem Datensatz
    '''
    train_dset = load_tfrecord_dataset(dataset_root_path + subfolders[0], 'train.record')
    val_dset = load_tfrecord_dataset(dataset_root_path + subfolders[0], 'val.record')

    for sub_folder in subfolders[1:]:
        dataset_path = dataset_root_path + sub_folder
        #print('Concat ' + sub_folder)
        train_temp = load_tfrecord_dataset(dataset_path, 'train.record')
        train_dset = train_dset.concatenate(train_temp)
        val_temp = load_tfrecord_dataset(dataset_path, 'val.record')
        val_dset = val_dset.concatenate(val_temp)

    #print('Nr. of Examples train: ' + str(sum(1 for _ in train_dset)))
    #print('Nr. of Examples validation: ' + str(sum(1 for _ in val_dset)))

    return train_dset, val_dset
                