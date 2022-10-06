"""Run experiment for tabular data.

This prints out a table summarizing the experiment and saves plots in desired
location.
"""

from typing import Callable, Any

from absl import app
import numpy as np

import interpretability_methods
import models
import tabular_experiment


def main(unused_argv: Any) -> None:
  """Program entry point."""
  recourse_experiment = tabular_experiment.TabularExperiment('wine_quality')
  spurious_experiment = tabular_experiment.TabularExperiment('wine_quality')

  def model_trainer(
      features_train: np.ndarray, labels_train: np.ndarray,
      features_test: np.ndarray,
      labels_test: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return models.tabular_classification(
        features_train,
        labels_train,
        features_test,
        labels_test,
        n_classes=recourse_experiment.n_classes,
        layer_sizes=[64] * 5,
        batch_size=500,
        epochs=30,
        verbose=2)

  basepath = 'YOUR/PATH'

  recourse_experiment.run_experiment(
      interpretability_methods.recourse_oracle,
      interpretability_methods.recourse_hypothesis_test,
      model_trainer,
      n_repetitions=3)
  recourse_experiment.visualize_experiment_table()
  recourse_experiment.visualize_hypothesis_test(
      'Wine Recourse Hypothesis Tradeoff',
      basepath + 'wine-recourse-hypothesis-tradeoff-cl')
  recourse_experiment.visualize_accuracy(
      'Wine Recourse Accuracy Tradeoff',
      basepath + 'wine-recourse-accuracy-tradeoff-cl')

  spurious_experiment.run_experiment(
      interpretability_methods.spurious_oracle,
      interpretability_methods.spurious_hypothesis_test,
      model_trainer,
      n_repetitions=3)
  spurious_experiment.visualize_experiment_table()
  spurious_experiment.visualize_hypothesis_test(
      'Wine Spurious Hypothesis Tradeoff',
      basepath + 'wine-spurious-hypothesis-tradeoff-cl')
  spurious_experiment.visualize_accuracy(
      'Wine Spurious Accuracy Tradeoff',
      basepath + 'wine-spurious-accuracy-tradeoff-cl')


if __name__ == '__main__':
  app.run(main)
