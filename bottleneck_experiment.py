"""Instantiate experiment object for high-dimensional data (e.g., images).

For tabular data, see tabular_experiment.py
When run, these methods perform the experiment designed specifically for
bottleneck models; i.e., the feature attribution is performed on the output of
an intermediate layer.
"""

import enum
from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds

import experiment
import interpretability_methods


class BottleneckDataset(enum.Enum):
  CIFAR10 = 'cifar10'


class BottleneckExperiment(experiment.Experiment):
  """Object for experiments with a low number of features."""

  def __init__(self, data: BottleneckDataset) -> None:
    super().__init__()
    if data == BottleneckDataset.CIFAR10:
      self._cifar10_data()
    else:
      raise ValueError('Dataset cannot be used for BottleneckExperiment.')

  def _cifar10_data(self) -> None:
    """Set data to be tensorflow wine_quality dataset."""

    self.n_data = 10000  # hardcoded, otherwise have to parse the whole dataset
    self.n_classes = 10

    data = tfds.load('cifar10', split=['train'])[0]

    # renormalize features to have mean=0 and sd=1
    data = tfds.as_dataframe(
        data.shuffle(buffer_size=self.n_data).take(self.n_data))
    features_data = np.stack(data['image'].values)
    self.features_data = np.array([
        (features_col - np.mean(features_col)) / np.std(features_col)
        for features_col in features_data.T
    ]).T
    self.labels_data = np.array(data['label']).astype(int)

  def run_experiment(self,
                     oracle: Callable[..., Any],
                     hypothesis_test: Callable[..., Any],
                     model: Callable[
                         [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
                         tf.keras.models.Sequential],
                     n_repetitions: int,
                     n_concepts: int,
                     n_train_samples: Optional[int] = None,
                     n_models: int = 1,
                     n_concept_samples: Optional[int] = None,
                     n_eval_samples: Optional[int] = None) -> None:
    """Runs the experiment using whatever the data has been set to for the object.

    Args:
      oracle: the function defining ground truth
      hypothesis_test: the function that converts feature attribution into a
        hypothesis test about the ground truth
      model: function to train the model. Accepts only the training and test
        data as arguments and the number of concepts (size of bottleneck layer);
        the other model parameters (layers, batch size, etc.) should be set
        before calling this method.
      n_repetitions: number of times to repeat the experiment, used to obtain
        error bars.
      n_concepts: number of dimensions that the input image is projected to for
        bottleneck layer.
      n_train_samples: number of data points to use for model fitting. If None
        then uses the whole dataset
      n_models: number of models to fit to the data
      n_concept_samples: number of conceptss to sample for hypothesis test
      n_eval_samples: number of examples to sample for hypothesis test
    """

    if n_train_samples is None:
      n_train_samples = self.n_data
    if n_concept_samples is None:
      n_concept_samples = np.min([5, n_concepts])
    if n_eval_samples is None:
      n_eval_samples = np.min([5, self.n_data])

    self.experiment_params.reset(n_repetitions)

    for repetition_idx in range(n_repetitions):

      sample_idxs = np.random.randint(0, self.n_data, n_train_samples)
      sample_features_data = self.features_data[sample_idxs]
      sample_labels_data = self.labels_data[sample_idxs]

      features_train, features_test, labels_train, labels_test = (
          train_test_split(
              sample_features_data, sample_labels_data, test_size=0.2))

      for _ in range(n_models):

        trained_model = model(features_train, labels_train, features_test,
                              labels_test, n_concepts)
        encoder_model = tf.keras.Model(
            inputs=trained_model.input,
            outputs=trained_model.get_layer('bottleneck').output)
        decoder_model = tf.keras.Model(
            inputs=trained_model.get_layer('bottleneck').output,
            outputs=trained_model.output)
        decoder_model_evaluated = lambda x: decoder_model(x).numpy()  # pylint: disable=cell-var-from-loop

        features_encoded = encoder_model(features_train).numpy()
        baseline = np.mean(features_encoded, axis=0)

        train_idxs = np.random.randint(
            0, len(features_train), size=n_eval_samples)
        for train_idx in train_idxs:
          example = features_encoded[train_idx]
          class_list = [
              np.argmax(decoder_model(example.reshape((1, n_concepts))))
          ]
          feature_list = np.array(range(n_concepts))
          oracle_value = oracle(decoder_model, example, class_list,
                                feature_list)
          shap_value = hypothesis_test(
              interpretability_methods.shap(decoder_model_evaluated, example,
                                            class_list, feature_list,
                                            features_encoded))
          lime_value = hypothesis_test(
              interpretability_methods.lime(
                  decoder_model, example, class_list, feature_list,
                  reg_param=2))

          intgrad_value = hypothesis_test(
              interpretability_methods.integrated_gradient(
                  decoder_model,
                  example,
                  class_list,
                  feature_list,
                  baseline,
                  num_iters=20))

          grad_value = hypothesis_test(
              interpretability_methods.gradient(decoder_model, example,
                                                class_list, feature_list))

          # how many 1's
          self.experiment_params.n_oracle_positive[repetition_idx] += np.sum(
              oracle_value)

          # when oracle == 1, count 1's
          self.experiment_params.n_shap_true_positive[repetition_idx] += np.sum(
              oracle_value * shap_value)
          self.experiment_params.n_lime_true_positive[repetition_idx] += np.sum(
              oracle_value * lime_value)
          self.experiment_params.n_intgrad_true_positive[
              repetition_idx] += np.sum(oracle_value * intgrad_value)
          self.experiment_params.n_grad_true_positive[repetition_idx] += np.sum(
              oracle_value * grad_value)

          # when oracle == 0, count 0's
          self.experiment_params.n_shap_true_negative[repetition_idx] += np.sum(
              (1 - oracle_value) * (1 - shap_value))
          self.experiment_params.n_lime_true_negative[repetition_idx] += np.sum(
              (1 - oracle_value) * (1 - lime_value))
          self.experiment_params.n_intgrad_true_negative[
              repetition_idx] += np.sum(
                  (1 - oracle_value) * (1 - intgrad_value))
          self.experiment_params.n_grad_true_negative[repetition_idx] += np.sum(
              (1 - oracle_value) * (1 - grad_value))

          self.experiment_params.n_experiment_samples[repetition_idx] += len(
              feature_list) * len(class_list)
