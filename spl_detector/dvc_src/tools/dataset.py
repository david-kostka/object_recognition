# Zweck:
# Hilfsfunktionen zum erstellen von TFRecords aus Simulator Trainingsdaten
# Erzeugung eines tf.data.Dataset Objektes basierend auf den TFRecord Files und Bilddaten
# Author: David Kostka
# Datum: 05.11.2020

# TODO: In Klasse convertieren, Generalisieren so dass die Struktur der labels.csv definiert werden kann aber der Rest gleich bleibt (selbe TFRecord Struktur)
# Config Datei lesen um an Dataset anpassbar zu sein
# Definierbares mapping von CSV Spaltenname zu TFRecord Feature name z.B. {filename:"name", xmin:"minX", ...}

import pandas as pd
import tensorflow as tf
import numpy as np
import os
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt

orig_size = (480, 640)
target_size = (60, 80)

def create_tf_example(row, img_path):
    filename = row.name.encode('utf8')

    img_raw = open(img_path + row.name, 'rb').read()

    xmins = row.minX / orig_size[1]
    xmaxs = row.maxX / orig_size[1]
    ymins = row.minY / orig_size[0]
    ymaxs = row.maxY / orig_size[0]

    classes = row.classes

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[xmins])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[xmaxs])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[ymins])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[ymaxs])),
        'image/object/class/label': tf.train.Feature(float_list=tf.train.FloatList(value=[classes]))
    }))
    return tf_example

def write_tfrecord(labels, img_path, output_path):
    writer = tf.io.TFRecordWriter(output_path)

    for label in labels.itertuples():
        tf_example = create_tf_example(label, img_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print("TFRecord '" + output_path + "' created")

def read_csv(path):
    return pd.read_csv(path, sep=r'\s*,\s*', engine='python')

def create_tfrecord_from_csv(csv_path, img_path, output_path):
    labels = read_csv(csv_path)

    write_tfrecord(labels, img_path, output_path)

def split_csv(csv_path, output_path, train_name, test_name, test_split=0.2):
    labels = read_csv(csv_path)
    labels = labels.sample(frac=1).reset_index(drop=True)

    count = len(labels)
    split_index = int(count - count * test_split)

    labels[:split_index].to_csv(output_path + train_name, index=False)
    labels[split_index:].to_csv(output_path + test_name, index=False)

def create_clean_csv(csv_path, img_path, output_path, nr_neg_labels_weight=0.3):
    raw_labels = pd.read_csv(csv_path, sep=r'\s*,\s*', index_col=None, engine='python')
    raw_labels = raw_labels[['name', 'minX', 'minY', 'maxX', 'maxY']]

    raw_labels['classes'] = 1
    labels_img_names = raw_labels['name'].unique()
    all_img_names = os.listdir(img_path)
    no_nao_img_names = np.setdiff1d(all_img_names, labels_img_names)
    negative_labels = pd.DataFrame(no_nao_img_names, columns=['name'])
    negative_labels['classes'] = 0
    negative_labels = negative_labels.sample(frac=1).reset_index(drop=True)
    raw_labels = raw_labels.append(negative_labels[:int(len(negative_labels) * nr_neg_labels_weight)])
    raw_labels = raw_labels[raw_labels.name != 'name']

    labels = raw_labels.fillna(0).groupby('name').filter(lambda x: len(x) == 1)
    labels = labels.sample(frac=1).reset_index(drop=True)

    return labels.to_csv(output_path, index=False)

def parse_tfrecord(tfrecord):
    IMAGE_FEATURE_MAP = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.float32)
    }

    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=1)
    x_train = tf.image.resize(x_train, target_size)

    y_train = [ x['image/object/bbox/xmin'],
                x['image/object/bbox/ymin'],
                x['image/object/bbox/xmax'],
                x['image/object/bbox/ymax'],
                x['image/object/class/label']
                ]

    return x_train, y_train

def load_tfrecord_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset