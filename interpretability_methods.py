"""Methods to compute feature attributions.

Compute:
SHAP
Integrated Gradients
LIME
Gradient

Also, convert the outputs of these to a hypothesis test for recourse and
spurious feature identification.
"""

from typing import Callable, Optional

import numpy as np
import shap as third_party_shap
import tensorflow as tf


def shap(model: Callable[[np.ndarray], np.ndarray],
         example: np.ndarray,
         class_list: list[int],
         feature_list: np.ndarray,
         interpret_data: np.ndarray,
         num_iters: int = 100,
         max_subsets: int = 500) -> np.ndarray:
  """Wrapper to approximately compute SHAP for a small number of features.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}
    interpret_data: global dataset to define background expectation, must have
      rows of length p
    num_iters: number of data points to approximate expectations with
    max_subsets: number of samples of the 2^p subsets needed for exact
      computation

  Returns:
    array of dimension (len(feature_list), len(class_list)) with the SHAP
    values for each (feature, class) pair
  """
  explainer = third_party_shap.KernelExplainer(
      model, interpret_data[np.random.randint(0, interpret_data.shape[0],
                                              num_iters)])
  shap_values = np.array(explainer.shap_values(example, nsamples=max_subsets)).T
  return np.array(
      [shap_values[feature, class_list] for feature in feature_list])


def lime(model: Callable[[np.ndarray], np.ndarray],
         example: np.ndarray,
         class_list: list[int],
         feature_list: np.ndarray,
         num_iters: int = 1000,
         local_radius: float = 0.1,
         reg_param: float = 2.) -> np.ndarray:
  """Function to approximately compute LIME for a medium number of features.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}
    num_iters: number of data points to approximate expectations with
    local_radius: standard deviation of Gaussian used to sample local
      perturbations
    reg_param: value used for L2 regularization in linear regression step

  Returns:
    array of dimension (len(feature_list), len(class_list)) with the LIME
    values for each (feature, class) pair
  """

  # Gaussian kernel approach (no truncation)
  features_local = (
      local_radius * np.random.randn(num_iters, len(example)) + example)
  labels_local = model(features_local)

  # L2 regularization
  features_local_reg = np.vstack((features_local, reg_param * np.ones(
      (1, len(example)))))
  labels_local_reg = np.vstack(
      (labels_local, np.zeros((1, labels_local.shape[1]))))

  theta = np.linalg.lstsq(features_local_reg, labels_local_reg, rcond=None)[0]

  return np.array([theta[feature, class_list] for feature in feature_list
                  ]).reshape((len(feature_list), len(class_list)))


def integrated_gradient(model: Callable[[np.ndarray], np.ndarray],
                        example: np.ndarray,
                        class_list: list[int],
                        feature_list: np.ndarray,
                        baseline: Optional[np.ndarray] = None,
                        num_iters: int = 20) -> np.ndarray:
  """Function to approximately compute IG for any number of features.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}
    baseline: example to use as baseline in feature attribution definition
    num_iters: number of steps to approximate path integral with

  Returns:
    array of dimension (len(feature_list), len(class_list)) with the IG
    values for each (feature, class) pair
  """
  if baseline is None:
    baseline = np.zeros((len(example),))
  interpolated_features = tf.convert_to_tensor(
      np.array([(1 - alpha) * baseline + alpha * example
                for alpha in np.linspace(0, 1, num_iters)]))

  def integrated_gradient_class_idx(class_idx: int) -> np.ndarray:
    with tf.GradientTape() as tape:
      tape.watch(interpolated_features)
      model_output = model(interpolated_features)[:, class_idx]
    gradients = tape.gradient(model_output, interpolated_features)
    return (example - baseline) * np.sum(gradients, axis=0) / num_iters

  return np.array([
      integrated_gradient_class_idx(class_idx)[feature_list]
      for class_idx in class_list
  ]).reshape((len(feature_list), len(class_list)))


def gradient(model: Callable[[np.ndarray], np.ndarray], example: np.ndarray,
             class_list: list[int], feature_list: np.ndarray) -> np.ndarray:
  """Function to exactly compute Gradient for any number of features.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}

  Returns:
    array of dimension (len(feature_list), len(class_list)) with the Gradient
    for each (feature, class) pair
  """
  example_tensor = tf.convert_to_tensor(example.reshape((1, len(example))))

  def gradient_class_idx(class_idx: int) -> np.ndarray:
    with tf.GradientTape() as tape:
      tape.watch(example_tensor)
      model_output = model(example_tensor)[:, class_idx]
    return tape.gradient(model_output, example_tensor).numpy().reshape(
        (len(example),))

  return np.array([
      gradient_class_idx(class_idx)[feature_list] for class_idx in class_list
  ]).reshape((len(feature_list), len(class_list)))


def recourse_oracle(model: Callable[[np.ndarray], np.ndarray],
                    example: np.ndarray,
                    class_list: list[int],
                    feature_list: np.ndarray,
                    delta: float = 1.) -> np.ndarray:
  """Function to compute the correct direction for recourse.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}
    delta: radius of interval to consider perturbing the feature over

  Returns:
    0/1 valued array of dimension (len(feature_list), len(class_list))
    0: decreasing feature increases class probability,
    1: increasing feature increases class probability
  """
  n_recourse_steps = 20
  delta_list = np.array([[delta, delta] for feature in feature_list])

  example_left = []
  example_right = []
  for feature_idx, feature in enumerate(feature_list):
    example_left_rep = np.tile(example, (n_recourse_steps, 1))
    example_left_rep[:, feature] = (
        np.linspace(example[feature] - delta_list[feature_idx, 0],
                    example[feature], n_recourse_steps))
    example_left.append(example_left_rep)

    example_right_rep = np.tile(example, (n_recourse_steps, 1))
    example_right_rep[:, feature] = (
        np.linspace(example[feature],
                    example[feature] + delta_list[feature_idx, 1],
                    n_recourse_steps))
    example_right.append(example_right_rep)

  mean_left = np.mean(
      np.array([
          model(example_left[feature_idx])
          for feature_idx in range(len(feature_list))
      ]),
      axis=1)[:, class_list]
  mean_right = np.mean(
      np.array([
          model(example_right[feature_idx])
          for feature_idx in range(len(feature_list))
      ]),
      axis=1)[:, class_list]

  return 1. * (mean_left < mean_right)


def spurious_oracle(model: Callable[[np.ndarray], np.ndarray],
                    example: np.ndarray,
                    class_list: list[int],
                    feature_list: np.ndarray,
                    delta: float = 1.,
                    quantile: float = 0.2) -> np.ndarray:
  """Function to compute whether model output is sensitive to features.

  Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: classes to get interpretation for, subset of {0,...,k-1}
    feature_list: features to get interpretation for, subset of {0,...,p-1}
    delta: radius of interval to consider perturbing the feature over
    quantile: determining factor of what constitutes "meaningful"

  Returns:
    0/1 valued array of dimension (len(feature_list), len(class_list))
    0: model output insensistive to feature,
    1: model output sensistive to feature
  """
  n_recourse_steps = 40
  delta_list = np.array([[delta, delta] for feature in feature_list])

  example_list = []
  for feature_idx, feature in enumerate(feature_list):
    example_rep = np.tile(example, (n_recourse_steps, 1))
    example_rep[:, feature] = (
        np.linspace(example[feature] - delta_list[feature_idx, 0],
                    example[feature] + delta_list[feature_idx, 1],
                    n_recourse_steps))
    example_list.append(example_rep)

  example_std = np.std(
      np.array([
          model(example_list[feature_idx])
          for feature_idx in range(len(feature_list))
      ]),
      axis=1)[:, class_list]

  return 1. * (example_std >= np.quantile(example_std, quantile))


def recourse_hypothesis_test(
    interpretability_method_output: np.ndarray) -> np.ndarray:
  """Function to convert interpretability method to recourse hypothesis test.

  Args:
    interpretability_method_output: the array of feature attribution from an
      interpretability method

  Returns:
    array of same shape as input, but thresholded for whether the attribution
    was positive or negative
  """
  return interpretability_method_output > 0


def spurious_hypothesis_test(interpretability_method_output: np.ndarray,
                             quantile: float = 0.2) -> np.ndarray:
  """Function to convert interpretability method to spurious features hypothesis test.

  Args:
    interpretability_method_output: the array of feature attribution from an
      interpretability method
    quantile: determining factor of what constitutes "meaningful"

  Returns:
    array of same shape as input, but thresholded for whether the absolute
    value of the attribution was (meaningfully) large
  """
  return 1. * (
      np.abs(interpretability_method_output) >= np.quantile(
          np.abs(interpretability_method_output), quantile))
