"""Run experiment for high-dimensional data with bottleneck model.

This prints out a table summarizing the experiment and saves plots in desired
location.
"""

from typing import Any

from absl import app
import numpy as np
import tensorflow as tf

import bottleneck_experiment
import interpretability_methods
import models


def main(unused_argv: Any) -> None:
  """Program entry point."""
  recourse_experiment = bottleneck_experiment.BottleneckExperiment(
      bottleneck_experiment.BottleneckDataset.CIFAR10)
  spurious_experiment = bottleneck_experiment.BottleneckExperiment(
      bottleneck_experiment.BottleneckDataset.CIFAR10)

  def model_trainer(features_train: np.ndarray, labels_train: np.ndarray,
                    features_test: np.ndarray, labels_test: np.ndarray,
                    n_concepts: int) -> tf.keras.models.Sequential:
    return models.bottleneck_classification(
        features_train,
        labels_train,
        features_test,
        labels_test,
        n_classes=recourse_experiment.n_classes,
        n_concepts=n_concepts,
        layer_sizes=[20] * 5,
        batch_size=32,
        epochs=5,
        verbose=2)

  basepath = 'YOUR/PATH'

  recourse_experiment.run_experiment(
      interpretability_methods.recourse_oracle,
      interpretability_methods.recourse_hypothesis_test,
      model_trainer,
      n_repetitions=10,
      n_concepts=8)
  recourse_experiment.visualize_experiment_table()
  recourse_experiment.visualize_hypothesis_test(
      'CIFAR10 Recourse Hypothesis Tradeoff',
      basepath + 'cifar10-recourse-hypothesis-tradeoff-cl')
  recourse_experiment.visualize_accuracy(
      'CIFAR10 Recourse Accuracy Tradeoff',
      basepath + 'cifar10-recourse-accuracy-tradeoff-cl')

  spurious_experiment.run_experiment(
      interpretability_methods.spurious_oracle,
      interpretability_methods.spurious_hypothesis_test,
      model_trainer,
      n_repetitions=10,
      n_concepts=8)
  spurious_experiment.visualize_experiment_table()
  spurious_experiment.visualize_hypothesis_test(
      'CIFAR10 Spurious Hypothesis Tradeoff',
      basepath + 'cifar10-spurious-hypothesis-tradeoff-cl')
  spurious_experiment.visualize_accuracy(
      'CIFAR10 Spurious Accuracy Tradeoff',
      basepath + 'cifar10-spurious-accuracy-tradeoff-cl')


if __name__ == '__main__':
  app.run(main)
