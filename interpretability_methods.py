"""Methods to compute feature attributions.

Compute:
SHAP
Integrated Gradients
LIME
Gradient
SmoothGrad
"""

from typing import Callable, Optional, List

import numpy as np
import shap as third_party_shap
import tensorflow as tf


def shap(model: Callable[[np.ndarray], np.ndarray],
         example: np.ndarray,
         pred_class: int,
         interpret_data: np.ndarray,
         num_iters: int = 100,
         max_subsets: int = 500) -> np.ndarray:
    """Wrapper to approximately compute SHAP for a small number of features.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: class to get interpretation for
    interpret_data: global dataset to define background expectation, must have
      rows of length p
    num_iters: number of data points to approximate expectations with
    max_subsets: number of samples of the 2^p subsets needed for exact
      computation

    Returns:
    array of shape matching features with the SHAP values
    """
    data_subset = interpret_data[np.random.randint(0, interpret_data.shape[0],num_iters)]
    explainer = third_party_shap.KernelExplainer(model, data_subset)
    shap_values = np.array(explainer.shap_values(example, nsamples=max_subsets)).T
    return shap_values[:, pred_class]

def image_shap(model: Callable[[np.ndarray], np.ndarray],
         example: np.ndarray,
         pred_class: int,
         num_iters: int = 1000,
         batch_size: int = 50) -> np.ndarray:
    """Wrapper to approximately compute Partition SHAP for image models.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    class_list: class to get interpretation for
    interpret_data: global dataset to define background expectation, must have
      rows of length p
    num_iters: number of data points to approximate expectations with
    batch_size: number of samples of the 2^p subsets needed for exact
      computation

    Returns:
    array of shape matching features with the SHAP values
    """
    masker_blur = third_party_shap.maskers.Image("blur(8,8)", example.shape)
    explainer = third_party_shap.Explainer(model, masker_blur)
    shap_values = explainer(example.reshape((1,) + example.shape), max_evals=num_iters, batch_size=batch_size)
    return shap_values.values[0,:,:,:,pred_class]


def lime(model: Callable[[np.ndarray], np.ndarray],
         example: np.ndarray,
         pred_class: int,
         num_iters: int = 100,
         local_radius: float = 0.1,
         reg_param: float = 1.) -> np.ndarray:
    """Function to approximately compute LIME for a medium number of features.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    pred_class: class to get interpretation for
    num_iters: number of data points to approximate expectations with
    local_radius: standard deviation of Gaussian used to sample local
      perturbations
    reg_param: value used for L2 regularization in linear regression step

    Returns:
    array of shape matching features with the LIME values
    """

    # Gaussian kernel approach (no truncation)
    features_local = (np.sqrt(local_radius) * np.random.randn(*((num_iters,) + example.shape)) + example)
    labels_local = model(features_local)[:, pred_class]

    # L2 regularization
    features_local_reg = np.vstack((features_local, reg_param * np.ones((1,) + example.shape)))
    labels_local_reg = np.hstack((labels_local, np.array([0])))

    theta = np.linalg.lstsq(features_local_reg.reshape(features_local_reg.shape[0], -1), labels_local_reg, rcond=None)[0]

    return theta.reshape(example.shape)


def integrated_gradient(model: Callable[[np.ndarray], np.ndarray],
                        example: np.ndarray,
                        pred_class: int,
                        baseline: Optional[np.ndarray] = None,
                        num_iters: int = 20) -> np.ndarray:
    """Function to approximately compute IG for any number of features.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    pred_class: class to get interpretation for
    baseline: example to use as baseline in feature attribution definition
    num_iters: number of steps to approximate path integral with

    Returns:
    array of shape matching features with the IG values
    """
    if baseline is None:
        baseline = np.zeros(example.shape)
    interpolated_features = tf.convert_to_tensor(
        np.array([(1 - alpha) * baseline + alpha * example for alpha in np.linspace(0, 1, num_iters)]))

    with tf.GradientTape() as tape:
        tape.watch(interpolated_features)
        model_output = model(interpolated_features)[:, pred_class]
    gradients = tape.gradient(model_output, interpolated_features)
    return (example - baseline) * np.mean(gradients, axis=0)


def gradient(model: Callable[[np.ndarray], np.ndarray],
             example: np.ndarray,
             pred_class: int) -> np.ndarray:
    """Function to exactly compute Gradient for any number of features.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    pred_class: class to get interpretation for

    Returns:
    array of shape matching features with the Gradient
    """
    example_tensor = tf.convert_to_tensor(example.reshape((1,)+example.shape))

    with tf.GradientTape() as tape:
        tape.watch(example_tensor)
        model_output = model(example_tensor)[:, pred_class]
    return tape.gradient(model_output, example_tensor).numpy().reshape(example.shape)

def smoothgrad(model: Callable[[np.ndarray], np.ndarray],
             example: np.ndarray,
             pred_class: int,
             num_iters: int = 100,
             local_radius: float = 0.1,) -> np.ndarray:
    """Function to exactly compute SmoothGrad for any number of features.

    Args:
    model: function that maps features (dimension p) to probabilities over
      classes (dimension k)
    example: an array of shape (p,)
    pred_class: class to get interpretation for
    num_iters: number of data points to approximate expectations with
    local_radius: standard deviation of Gaussian used to sample local
      perturbations

    Returns:
    array of shape matching features with the SmoothGrad
    """
    features_local = tf.convert_to_tensor(np.sqrt(local_radius) * np.random.randn(*((num_iters,) + example.shape)) + example)
    with tf.GradientTape() as tape:
        tape.watch(features_local)
        model_output = model(features_local)[:, pred_class]
    gradients = tape.gradient(model_output, features_local)
    return np.mean(gradients, axis=0)
