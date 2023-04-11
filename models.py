"""Model architectures to be used in experiments.

n_features and n_classes are hardcoded in.
Minimal finetuning on architecture was done.
"""

from typing import Optional

import numpy as np
import tensorflow as tf
import datasets


def create_tabular_model(data: datasets.TabularDataset):
    if data == datasets.TabularDataset.WINE:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(12, input_dim=13))

            model.add(tf.keras.layers.Dense(24, activation='relu'))

            model.add(tf.keras.layers.Dense(8, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(3, activation='softmax'))

            #opt = tf.keras.optimizers.SGD(lr=0.1)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

            return model

    elif data == datasets.TabularDataset.ABALONE:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(256, input_dim=8, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())


            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Dense(1, activation='relu'))

            #opt = tf.keras.optimizers.SGD(lr=0.1)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mean_squared_error'])

            return model

    elif data == datasets.TabularDataset.CREDIT:

        def create_model():
            model = tf.keras.models.Sequential()

            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(256, input_dim=15))

            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dense(128, activation='relu'))

            #model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))

            #model.add(tf.keras.layers.Dense(8, activation='relu'))

            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            #opt = tf.keras.optimizers.SGD(lr=0.1)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

            return model

    elif data == datasets.TabularDataset.CHESS:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(256, input_dim=6))

            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            #opt = tf.keras.optimizers.SGD(lr=0.1)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

            return model

    elif data == datasets.TabularDataset.ECOLI:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(12, input_dim=7))
            model.add(tf.keras.layers.Dense(24, activation='relu'))

            model.add(tf.keras.layers.Dense(8, activation='softmax'))

            #opt = tf.keras.optimizers.SGD(lr=0.9)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

            return model

    else:
        raise ValueError('Dataset cannot be used for TabularExperiment.')

    return create_model

def create_image_model(data: datasets.ImageDataset):
    if data == datasets.ImageDataset.CIFAR10:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Flatten())

            model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

            return model

    elif data == datasets.ImageDataset.MNIST:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28,28,1)))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dropout(0.5))

            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

            return model

    elif data == datasets.ImageDataset.FASHION:

        def create_model():
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28,28,1)))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dropout(0.5))

            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

            return model

    else:
        raise ValueError('Dataset cannot be used for ImageExperiment.')

    return create_model
