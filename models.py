"""Model architectures to be used in experiments.

Currently, focusing on models for tabular data.
"""

from typing import Optional

import numpy as np
import tensorflow as tf


def tabular_classification(features_train: np.ndarray,
                           labels_train: np.ndarray,
                           features_test: np.ndarray,
                           labels_test: np.ndarray,
                           n_classes: int,
                           layer_sizes: Optional[list[int]] = None,
                           batch_size: Optional[int] = None,
                           epochs: int = 30,
                           verbose: int = 2) -> tf.keras.models.Sequential:
  """Fits a fully connected neural network for classification.

  Args:
    features_train: Array of features to train the model
    labels_train: Array of labels to train the model. Labels should be integers,
      rather than one-hot.
    features_test: Array of features to validate the model
    labels_test: Array of labels to validate the model
    n_classes: number of classes for classification. Hardcoded to avoid errors
      where certain classes aren't in data.
    layer_sizes: Either an integer or a list of integers (the latter if
      different number of nodes desired per layer)
    batch_size: Batch size
    epochs: Number of training epochs
    verbose: input inherited from tf.keras.models.Sequential.fit
      0: don't print output during training
      2: print output during training

  Returns:
    trained Sequential model
  """
  model = tf.keras.models.Sequential()
  input_dim = features_train.shape[1]
  output_dim = n_classes
  if layer_sizes is None:
    layer_sizes = [20] * 3
  n_layers = len(layer_sizes)
  if batch_size is None:
    batch_size = np.max([int(0.1 * features_train.shape[0]), 1])

  model.add(
      tf.keras.layers.Dense(
          layer_sizes[0], input_dim=input_dim, name='input_mapping'))

  for l in range(n_layers - 1):
    model.add(
        tf.keras.layers.Dense(
            layer_sizes[l + 1], activation='relu', name='relu%s' % l))
    model.add(tf.keras.layers.Dense(layer_sizes[l + 1], name='fc%s' % l))

  model.add(
      tf.keras.layers.Dense(output_dim, name='final', activation='softmax'))

  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])
  if verbose != 0:
    print('input dimension: %s ' % input_dim)
    model.summary(print_fn=print)

  model.fit(
      features_train,
      labels_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=verbose,
      validation_data=(features_test, labels_test))

  return model


def bottleneck_classification(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    n_classes: int,
    n_concepts: int = 10,
    layer_sizes: Optional[list[int]] = None,
    batch_size: Optional[int] = None,
    epochs: int = 30,
    verbose: int = 2) -> tf.keras.models.Sequential:
  """Fits a fully connected neural network for classification.

  Args:
    features_train: Array of features to train the model
    labels_train: Array of labels to train the model. Labels should be integers,
      rather than one-hot.
    features_test: Array of features to validate the model
    labels_test: Array of labels to validate the model
    n_classes: number of classes for classification. Hardcoded to avoid errors
      where certain classes aren't in data.
    n_concepts: size of intermediate layer that will be the "features" on which
      interpretability methods are run
    layer_sizes: Either an integer or a list of integers (the latter if
      different number of nodes desired per layer)
    batch_size: Batch size
    epochs: Number of training epochs
    verbose: input inherited from tf.keras.models.Sequential.fit
      0: don't print output during training
      2: print output during training

  Returns:
    trained Sequential model
  """
  model = tf.keras.models.Sequential()
  input_dim = features_train.shape[1:]
  output_dim = n_classes
  if layer_sizes is None:
    layer_sizes = [20] * 3
  n_layers = len(layer_sizes)
  if batch_size is None:
    batch_size = np.max([int(0.1 * features_train.shape[0]), 1])

  model = tf.keras.models.Sequential()
  model.add(
      tf.keras.layers.Conv2D(
          32, (3, 3), activation='relu', padding='same', input_shape=input_dim))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(
      tf.keras.layers.Conv2D(
          32,
          (3, 3),
          activation='relu',
          padding='same',
      ))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(
      tf.keras.layers.Conv2D(
          64,
          (3, 3),
          activation='relu',
          padding='same',
      ))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(
      tf.keras.layers.Conv2D(
          64,
          (3, 3),
          activation='relu',
          padding='same',
      ))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(
      tf.keras.layers.Conv2D(
          128,
          (3, 3),
          activation='relu',
          padding='same',
      ))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(
      tf.keras.layers.Conv2D(
          128,
          (3, 3),
          activation='relu',
          padding='same',
      ))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))

  model.add(
      tf.keras.layers.Dense(
          layer_sizes[0], input_dim=input_dim, name='input_mapping'))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(
      tf.keras.layers.Dense(n_concepts, activation='relu', name='bottleneck'))

  for l in range(n_layers - 1):
    model.add(
        tf.keras.layers.Dense(
            layer_sizes[l + 1], activation='relu', name='relu%s' % l))
    model.add(tf.keras.layers.Dense(layer_sizes[l + 1], name='fc%s' % l))

  model.add(
      tf.keras.layers.Dense(output_dim, name='final', activation='softmax'))

  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])
  if verbose != 0:
    model.summary(print_fn=print)

  model.fit(
      features_train,
      labels_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=verbose,
      validation_data=(features_test, labels_test))

  return model
