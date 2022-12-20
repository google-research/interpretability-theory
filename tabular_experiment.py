"""Instantiate experiment object for tabular data.

This file runs the experiment with code designed specifically for tabular data;
i.e., the feature attribution is performed directly on the model.
"""

import enum
from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

import experiment
import interpretability_methods


class TabularDataset(enum.Enum):
  WINE_QUALITY = 'wine_quality'


class TabularExperiment(experiment.Experiment):
  """Object for experiments with a low number of features."""

  def __init__(self, data: TabularDataset) -> None:
    super().__init__()
    if data == TabularDataset.WINE_QUALITY:
      self._wine_quality_data()
    else:
      raise ValueError('Dataset cannot be used for TabularExperiment.')

  def _wine_quality_data(self) -> None:
    """Set data to be tensorflow wine_quality dataset."""

    feature_names = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
    ]
    self.n_data = 4898  # hardcoded, otherwise have to parse the whole dataset
    self.n_features = len(feature_names)
    self.n_classes = 7

    data = tfds.load('wine_quality', split='train')

    data = np.array(tfds.as_dataframe(data.take(self.n_data)))
    features_data = data[:, 0:self.n_features]
    self.features_data = np.array([
        (features_col - np.mean(features_col)) / np.std(features_col)
        for features_col in features_data.T
    ]).T
    self.labels_data = data[:, self.n_features] - 3

  def run_experiment(self,
                     oracle: Callable[..., Any],
                     hypothesis_test: Callable[..., Any],
                     model: Callable[
                         [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         Callable[[np.ndarray], np.ndarray]],
                     n_repetitions: int = 1,
                     n_train_samples: Optional[int] = None,
                     n_models: int = 1,
                     n_feature_samples: Optional[int] = None,
                     n_eval_samples: Optional[int] = None) -> None:
    """Runs the experiment using whatever the data has been set to for the object.

    Args:
      oracle: the function defining ground truth
      hypothesis_test: the function that coverts feature attribution into a
        hypothesis test about the ground truth
      model: function to train the model. Accepts only the training and test
        data as arguments; the model parameters (layers, batch size, etc.)
        should be set before calling this method.
      n_repetitions: number of times to repeat the experiment, used to obtain
        error bars.
      n_train_samples: number of data points to use for model fitting. If None
        then uses the whole dataset
      n_models: number of models to fit to the data
      n_feature_samples: number of features to sample for hypothesis test
      n_eval_samples: number of examples to sample for hypothesis test
    """

    if n_train_samples is None:
      n_train_samples = self.n_data
    if n_feature_samples is None:
      n_feature_samples = np.min([5, self.n_features])
    if n_eval_samples is None:
      n_eval_samples = np.min([5, self.n_data])

    self.experiment_params.reset(n_repetitions)

    for repetition_idx in range(n_repetitions):
      
      # used only for spurious features, but hackily pass it to both tests
      spurious_feature_quantile = np.random.uniform(low=0.1, high=0.8)

      sample_idxs = np.random.randint(0, self.n_data, n_train_samples)
      sample_features_data = self.features_data[sample_idxs]
      sample_labels_data = self.labels_data[sample_idxs]

      features_train, features_test, labels_train, labels_test = (
          train_test_split(
              sample_features_data, sample_labels_data, test_size=0.2))

      for _ in range(n_models):

        trained_model = model(features_train, labels_train, features_test,
                              labels_test)
        trained_model_evaluated = lambda x: trained_model(x).numpy()  # pylint: disable=cell-var-from-loop

        train_idxs = np.random.randint(
            0, len(features_train), size=n_eval_samples)
        for train_idx in train_idxs:
          example = features_train[train_idx]
          class_list = [
              np.argmax(trained_model(example.reshape((1, self.n_features))))
          ]
          feature_list = np.array(range(self.n_features))
          oracle_value = oracle(trained_model, example, class_list,
                                feature_list, spurious_feature_quantile=spurious_feature_quantile)
          shap_value = hypothesis_test(
              interpretability_methods.shap(trained_model_evaluated, example,
                                            class_list, feature_list,
                                            self.features_data), spurious_feature_quantile)
          lime_value = hypothesis_test(
              interpretability_methods.lime(
                  trained_model, example, class_list, feature_list,
                  reg_param=0), spurious_feature_quantile)

          intgrad_value = hypothesis_test(
              interpretability_methods.integrated_gradient(
                  trained_model,
                  example,
                  class_list,
                  feature_list,
                  num_iters=20), spurious_feature_quantile)

          grad_value = hypothesis_test(
              interpretability_methods.gradient(trained_model, example,
                                                class_list, feature_list), spurious_feature_quantile)

          # how many 1's
          self.experiment_params.n_oracle_positive[repetition_idx] += np.sum(
              oracle_value)

          # when oracle == 1, count 1's
          self.experiment_params.n_shap_true_positive[
              repetition_idx] += np.sum(oracle_value * shap_value)
          self.experiment_params.n_lime_true_positive[
              repetition_idx] += np.sum(oracle_value * lime_value)
          self.experiment_params.n_intgrad_true_positive[
              repetition_idx] += np.sum(oracle_value * intgrad_value)
          self.experiment_params.n_grad_true_positive[
              repetition_idx] += np.sum(oracle_value * grad_value)

          # when oracle == 0, count 0's
          self.experiment_params.n_shap_true_negative[
              repetition_idx] += np.sum((1 - oracle_value) * (1 - shap_value))
          self.experiment_params.n_lime_true_negative[
              repetition_idx] += np.sum((1 - oracle_value) * (1 - lime_value))
          self.experiment_params.n_intgrad_true_negative[
              repetition_idx] += np.sum(
                  (1 - oracle_value) * (1 - intgrad_value))
          self.experiment_params.n_grad_true_negative[
              repetition_idx] += np.sum((1 - oracle_value) * (1 - grad_value))

          self.experiment_params.n_experiment_samples[repetition_idx] += len(
              feature_list) * len(class_list)
