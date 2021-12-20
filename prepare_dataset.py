import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import rasterio

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_root_dir')
    parser.add_argument('single_label')
    parser.add_argument('output_directory')
    parser.add_argument('datasize')
    parser.add_argument('pixel_size')
    return parser.parse_args()


def get_image_folders(path):
    return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]


def load_labels(filename):
    if os.path.isfile(filename):
        df = pd.read_csv(filename, sep=r',', engine='python')
        return df.rename(columns={'IMAGE\LABEL': 'label'}).set_index('label')
    else:
        raise ValueError('The given file does not exist!')


def get_random_splits(datasize):
    indices = np.linspace(0, datasize, datasize, dtype=int)
    train_indices, test_indices = train_test_split(indices, test_size=0.2)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25)
    return train_indices, test_indices, val_indices


def get_example(band1, band2, band3, label, name):
    return tf.train.Example(features=tf.train.Features(feature={
        'B1': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(band1))),
        'B2': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(band2))),
        'B3': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(band3))),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        'patch_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')]))
    }))


if __name__ == '__main__':
    args = parse_args()
    folders = get_image_folders(args.image_root_dir)
    labels = load_labels(args.single_label)
    datasize = int(args.datasize)
    pixel_size = int(args.pixel_size)
    train_indices, test_indices, val_indices = get_random_splits(datasize)

    train_writer = tf.io.TFRecordWriter(os.path.join(args.output_directory, 'train.tfrecord'))
    val_writer = tf.io.TFRecordWriter(os.path.join(args.output_directory, 'val.tfrecord'))
    test_writer = tf.io.TFRecordWriter(os.path.join(args.output_directory, 'test.tfrecord'))

    progress_bar = tf.keras.utils.Progbar(datasize, unit_name='image')
    progress_bar.update(0)
    counter = 0
    val, test, train = 0, 0, 0

    for folder in folders:
        pathname = os.path.join(args.image_root_dir, folder)
        files = [f for f in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, f))]

        for file in files:
            band_ds = rasterio.open(os.path.join(pathname, file))
            band1 = tf.image.resize(tf.expand_dims(band_ds.read(1), -1), (pixel_size, pixel_size))
            band2 = tf.image.resize(tf.expand_dims(band_ds.read(2), -1), (pixel_size, pixel_size))
            band3 = tf.image.resize(tf.expand_dims(band_ds.read(3), -1), (pixel_size, pixel_size))
            example = get_example(band1, band2, band3, labels.loc[file[:-4]].values, file[:-4])

            if counter in train_indices:
                train_writer.write(example.SerializeToString())
                train += 1
            elif counter in test_indices:
                test_writer.write(example.SerializeToString())
                test += 1
            elif counter in val_indices:
                val_writer.write(example.SerializeToString())
                val += 1

            counter += 1
            progress_bar.add(1)

    print(f'Total scanned images = {test + val + train}')
    print(f'Train images = {train}')
    print(f'Test images = {test}')
    print(f'Val images = {val}')

    train_writer.close()
    test_writer.close()
    val_writer.close()
