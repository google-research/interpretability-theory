"""Tests for computing summaries and visualizing experiments.

Currently, just testing table output.
"""

import numpy as np
import experiment

def test_linear_function(self) -> None:
  """Test visualizing table output.

  Visually inspect that the output makes sense.
  The numbers are randomly generated, but they should all be probabilities.
  """

  experiment_obj = experiment.Experiment()
  experiment_obj.experiment_params.reset(1)

  # Use for testing. Sets the values to positive real numbers.
  # Experiment scale is arbitrary.
  # Tests for case when n_repetitions=1
  n_experiments_scale = 20
  random_values = np.random.uniform(
      low=1, high=n_experiments_scale, size=8).astype(int)

  experiment_obj.experiment_params.n_shap_true_negative[0] = random_values[
      0]
  experiment_obj.experiment_params.n_lime_true_negative[0] = random_values[
      1]
  experiment_obj.experiment_params.n_intgrad_true_negative[
      0] = random_values[2]
  experiment_obj.experiment_params.n_grad_true_negative[0] = random_values[
      3]
  experiment_obj.experiment_params.n_shap_true_positive[0] = random_values[
      4]
  experiment_obj.experiment_params.n_lime_true_positive[0] = random_values[
      5]
  experiment_obj.experiment_params.n_intgrad_true_positive[
      0] = random_values[6]
  experiment_obj.experiment_params.n_grad_true_positive[0] = random_values[
      7]
  experiment_obj.experiment_params.n_oracle_positive[
      0] = np.random.uniform(
          low=np.max(random_values[0:4]),
          high=2 * n_experiments_scale,
          size=1).astype(int)[0]
  experiment_obj.experiment_params.n_experiment_samples[
      0] = np.random.uniform(
          low=2 * np.max(random_values), high=3 * n_experiments_scale,
          size=1).astype(int)[0]

  # Check that accuracy is a convex combination of
  # specificity and sensitivity.
  shap_sensitivity = experiment_obj._sensitivity(
      experiment_obj.experiment_params.n_shap_true_positive)[0]
  shap_specificity = experiment_obj._specificity(
      experiment_obj.experiment_params.n_shap_true_negative)[0]
  shap_accuracy = experiment_obj._accuracy(
      experiment_obj.experiment_params.n_shap_true_positive,
      experiment_obj.experiment_params.n_shap_true_negative)[0]

  shap_accuracy < = np.max([shap_sensitivity, shap_specificity])
  shap_accuracy >= np.min([shap_sensitivity, shap_specificity])

  experiment_obj.visualize_experiment_table()

