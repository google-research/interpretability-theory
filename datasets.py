"""Datasets for interpretability theory experiments.

This class is used to load in data and store relevant information, such as which features are non-categorical.
"""

import enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import tensorflow as tf

data_basepath = 'saved-data/datasets/'


class TabularDataset(enum.Enum):
    WINE = 'wine'
    ABALONE = 'abalone'
    CREDIT = 'credit'
    CHESS = 'chess'
    ECOLI = 'ecoli'


class ImageDataset(enum.Enum):
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'
    FASHION = 'fashion'


def import_tabular_dataset(data: TabularDataset):
    """Standardize data must preserve the shape for tabular data."""

    if data == TabularDataset.WINE:
        # https://archive.ics.uci.edu/ml/datasets/Wine

        ordered_feature_idxs = [i for i in range(13)]

        data = pd.read_csv(data_basepath + 'wine.csv')
        data = data.sample(frac=1)

        labels_data = np.array(data['Cultivars'] - 1)
        features_data = np.array(data.drop('Cultivars', axis=1))[:,1:14]

        feature_mean = np.mean(features_data,axis=0)
        feature_std = np.std(features_data,axis=0)

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return (data - feature_mean) / feature_std

    elif data == TabularDataset.ABALONE:
        # https://archive.ics.uci.edu/ml/datasets/Abalone

        ordered_feature_idxs = [1,2,3,4,5,6,7]

        data = pd.read_csv(data_basepath + 'abalone.csv')
        data = data.sample(frac=1)

        data['Sex'] = data['Sex'].replace(['M','F','I'], [1,-1,0])

        labels_data = np.array(data['Rings'])
        features_data = np.array(data.drop('Rings',axis=1))

        feature_mean = np.mean(features_data,axis=0)
        feature_std = np.std(features_data,axis=0)

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return (data - feature_mean) / feature_std

    elif data == TabularDataset.CREDIT:
        # https://archive.ics.uci.edu/ml/datasets/Credit+Approval

        ordered_feature_idxs = [1,2,7,10,13,14]

        data = pd.read_csv(data_basepath + 'credit.csv', header=None)
        data = data.sample(frac=1)
        data = data.replace(to_replace='?', value=np.nan)

        # set float columns to float
        for float_col in [1,13]:
            data[float_col] = pd.to_numeric(data[float_col])

        # categorical replace with most common
        for categorical_col in [0,3,4,5,6]:
            data[categorical_col] = data[categorical_col].replace(np.nan, data[categorical_col].mode()[0])

        # float replace with mean
        for float_col in [1,13]:
            data[float_col] = data[float_col].replace(np.nan, data[float_col].mean())

        # make categorical into ints
        for categorical_col in [0,3,4,5,6,8,9,11,12,15]:
            data[categorical_col] = pd.factorize(data[categorical_col])[0]

        labels_data = np.array(data[15])
        features_data = np.array(data.drop(15,axis=1))

        feature_mean = np.mean(features_data,axis=0)
        feature_std = np.std(features_data,axis=0)

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return (data - feature_mean) / feature_std

    elif data == TabularDataset.CHESS:

        ordered_feature_idxs = [i for i in range(6)]

        data = pd.read_csv(data_basepath + 'chess.csv')
        data = data.sample(frac=1)

        # make categorical into ints
        for categorical_col in [0,2,4]:
            data.iloc[:, categorical_col] = pd.factorize(data.iloc[:,categorical_col])[0]

        # make labels binary
        data.iloc[:,6] = np.array([0 if val == 'draw' else 1 for val in data.iloc[:,6]])

        labels_data = np.array(data.iloc[:,6])
        features_data = np.array(data.iloc[: , :-1])

        feature_mean = np.mean(features_data,axis=0)
        feature_std = np.std(features_data,axis=0)

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return (data - feature_mean) / feature_std

    elif data == TabularDataset.ECOLI:
        # https://archive.ics.uci.edu/ml/datasets/Ecoli

        ordered_feature_idxs = [i for i in range(7)]

        data = pd.read_csv(data_basepath + 'ecoli.csv', header=None, sep="\s+")
        data = data.sample(frac=1)
        data = data.iloc[:,1:]

        # make labels ints
        data[8] = pd.factorize(data[8])[0]

        labels_data = np.array(data[8])
        features_data = np.array(data.drop(8,axis=1))

        feature_mean = np.mean(features_data,axis=0)
        feature_std = np.std(features_data,axis=0)

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return (data - feature_mean) / feature_std

    else:
        raise ValueError('Dataset cannot be used for TabularExperiment.')

    return features_data, labels_data, ordered_feature_idxs, standardize_data


def import_image_dataset(data: ImageDataset):
    """Standardize data must preserve the shape for image data."""

    if data == ImageDataset.CIFAR10:
        # https://www.tensorflow.org/datasets/catalog/cifar10

        (train_generator, test_generator), data_info = tfds.load(
            'cifar10',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        data_generator = tfds.load('cifar10', split=['test'])[0]
        data = tfds.as_dataframe(data_generator.shuffle(buffer_size=500).take(500))
        features_data = np.stack(data['image'].values)
        labels_data = np.array(data['label']).astype(int)

        def tf_standardize_data(image, label):
            return tf.cast(image, tf.float32) / 255., label

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return data / 255.

    elif data == ImageDataset.MNIST:
        # https://www.tensorflow.org/datasets/catalog/mnist

        (train_generator, test_generator), data_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        data_generator = tfds.load('mnist', split=['test'])[0]
        data = tfds.as_dataframe(data_generator.shuffle(buffer_size=500).take(500))
        features_data = np.stack(data['image'].values)
        labels_data = np.array(data['label']).astype(int)

        def tf_standardize_data(image, label):
            return tf.cast(image, tf.float32) / 255., label

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return data / 255.

    elif data == ImageDataset.FASHION:
        # https://www.tensorflow.org/datasets/catalog/fashion_mnist

        (train_generator, test_generator), data_info = tfds.load(
            'fashion_mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        data_generator = tfds.load('fashion_mnist', split=['test'])[0]
        data = tfds.as_dataframe(data_generator.shuffle(buffer_size=500).take(500))
        features_data = np.stack(data['image'].values)
        labels_data = np.array(data['label']).astype(int)

        def tf_standardize_data(image, label):
            return tf.cast(image, tf.float32) / 255., label

        def standardize_data(data: np.ndarray) -> np.ndarray:
            return data / 255.

    else:
        raise ValueError('Dataset cannot be used for ImageExperiment.')

    return features_data, labels_data, standardize_data, tf_standardize_data, train_generator, test_generator, data_info

