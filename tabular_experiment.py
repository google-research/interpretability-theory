"""Instantiate experiment object for tabular data.

This file runs the experiment with code designed specifically for tabular data;
i.e., the number of features is relatively small.
"""

import enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

import experiment
import interpretability_methods
import datasets
import models

model_basepath = 'saved-data/models/'
data_basepath = 'saved-data/experiments/'


class TabularExperiment(experiment.Experiment):
    """Object for experiments with a low number of features."""
    features_data: np.ndarray
    labels_data: np.ndarray
    ordered_feature_idxs: list[int]
    standardize_data: Callable[[np.ndarray], np.ndarray]

    def __init__(self, data: datasets.TabularDataset) -> None:
        super().__init__(data.value)
        self.features_data, self.labels_data, self.ordered_features_idxs, self.standardize_data = datasets.import_tabular_dataset(data)
        self.model_architecture = models.create_tabular_model(data)


    def train_models(self, n_models: int, epochs: int, batch_size: int) -> None:
        """Train tabular models where whole dataset can be stored in memory."""
        for model_idx in range(len(self.trained_models), len(self.trained_models) + n_models):
            model = self.model_architecture()
            model.summary(print_fn=print)

            features_train, features_test, labels_train, labels_test = (
                train_test_split(self.standardize_data(self.features_data),
                                 self.labels_data, test_size=0.2))

            history = model.fit(
                features_train,
                labels_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(features_test, labels_test))

            history_df = pd.DataFrame(history.history)
            with open(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx) + '_history.csv', mode='w') as f:
                history_df.to_csv(f)
            self.training_history.append(history_df)

            model.save_weights(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx))
            self.trained_models.append(model)

    def perturbation(self, example: np.ndarray, feature: int) -> np.ndarray:
        """Perturbations are always done on the actual scale, and then standardized to match the NN."""

        # hardcoded constants that will change interpretation
        n_steps = 20 # this should be even
        percent_perturb = 0.1

        value_range = np.max(self.features_data[:,feature]) - np.min(self.features_data[:,feature])
        perturbations = np.linspace(start=-percent_perturb*value_range, stop=percent_perturb*value_range, num=n_steps)

        perturbed_example = np.tile(example,(n_steps,1))
        perturbed_example[:,feature] = example[feature] + perturbations

        return perturbed_example

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """ Used to normalize the outputs of interpretability methods."""
        min_val = np.min(values)
        max_val = np.max(values)
        return 2*(values - min_val) / np.max(np.array([(max_val-min_val),1e-10])) - 1


    def run_experiment(self, experiment_type: experiment.ExperimentType, n_thresholds: int = 1, n_examples: int = 10) -> None:
        if n_thresholds == 1:
            threshold_vals = [0]
        elif n_thresholds < 10:
            threshold_vals = np.linspace(start=-n_thresholds/10,stop=n_thresholds/10,num=n_thresholds)
        else:
            threshold_vals = np.linspace(start=-1.0,stop=1.0,num=n_thresholds)
        if experiment_type == experiment.ExperimentType.RECOURSE:
            self.recourse_experiment_params.reset(len(self.trained_models) * len(threshold_vals))
        elif experiment_type == experiment.ExperimentType.SPURIOUS:
            self.spurious_experiment_params.reset(len(self.trained_models) * len(threshold_vals))
        else:
            raise ValueError('Unknown ExperimentType.')

        # each model uses the same examples
        examples = self.features_data[np.random.randint(0, self.features_data.shape[0], size=n_examples)]
        n_spurious_examples = 100
        spurious_examples =  self.features_data[np.random.randint(0, self.features_data.shape[0], size=n_spurious_examples)]

        # do each model separately
        for trained_model_idx, trained_model in enumerate(self.trained_models):

            def trained_model_evaluated(x):
                return trained_model(x).numpy()

            # for spurious features only, compute the model's quantile
            # note this means the oracle depends on the model
            if experiment_type == experiment.ExperimentType.SPURIOUS:
                spurious_variances = []
                for spurious_example_idx in range(n_spurious_examples):
                    spurious_example = spurious_examples[spurious_example_idx]
                    spurious_class = np.argmax(trained_model(self.standardize_data(spurious_example.reshape((1,)+spurious_example.shape))))
                    for spurious_feature in self.ordered_features_idxs:
                        perturbed_spurious_example = self.perturbation(spurious_example, spurious_feature)
                        perturbed_output = trained_model_evaluated(self.standardize_data(np.array(perturbed_spurious_example)))[:,spurious_class]
                        spurious_variances.append(np.var(perturbed_output))

                spurious_quantile = np.quantile(spurious_variances, 0.8)

            for example_idx in range(n_examples):

                # new images for each model
                example = examples[example_idx]
                max_class = np.argmax(trained_model(self.standardize_data(example.reshape((1,)+example.shape))))

                shap_values = interpretability_methods.shap(
                        trained_model_evaluated,
                        self.standardize_data(example),
                        max_class,
                        self.standardize_data(self.features_data))
                shap_values = self._normalize(shap_values)

                lime_values = interpretability_methods.lime(
                        trained_model,
                        self.standardize_data(example),
                        max_class)
                lime_values = self._normalize(lime_values)

                intgradA_values = interpretability_methods.integrated_gradient(
                        trained_model,
                        self.standardize_data(example),
                        max_class,
                        np.zeros(example.shape))
                intgradA_values = self._normalize(intgradA_values)

                intgradB_values = interpretability_methods.integrated_gradient(
                        trained_model,
                        self.standardize_data(example),
                        max_class,
                        np.min(self.standardize_data(self.features_data), axis=0))
                intgradB_values = self._normalize(intgradB_values)

                smoothgrad_values = interpretability_methods.smoothgrad(
                        trained_model,
                        self.standardize_data(example),
                        max_class)
                smoothgrad_values = self._normalize(smoothgrad_values)

                grad_values = interpretability_methods.gradient(
                        trained_model,
                        self.standardize_data(example),
                        max_class)
                grad_values = self._normalize(grad_values)

                # use all the features
                for feature in self.ordered_features_idxs:

                    perturbed_example = self.perturbation(example, feature)
                    perturbed_output = trained_model_evaluated(self.standardize_data(np.array(perturbed_example)))[:,max_class]

                    if experiment_type == experiment.ExperimentType.RECOURSE:
                        oracle_value = 1. * (np.mean(perturbed_output[int(len(perturbed_output)/2):len(perturbed_output)]) > np.mean(perturbed_output[0:int(len(perturbed_output)/2)]))
                    elif experiment_type == experiment.ExperimentType.SPURIOUS:
                        oracle_value = 1. * (np.var(perturbed_output) > spurious_quantile)
                    else:
                        raise ValueError('Unknown ExperimentType.')

                    for threshold_idx, threshold in enumerate(threshold_vals):

                        repetition_idx = trained_model_idx * len(threshold_vals) + threshold_idx

                        if experiment_type == experiment.ExperimentType.RECOURSE:
                            shap_value = 1. * (shap_values[feature] > threshold)
                            lime_value = 1. * (lime_values[feature] > threshold)
                            intgradA_value = 1. * (intgradA_values[feature] > threshold)
                            intgradB_value = 1. * (intgradB_values[feature] > threshold)
                            smoothgrad_value = 1. * (smoothgrad_values[feature] > threshold)
                            grad_value = 1. * (grad_values[feature] > threshold)

                            # how many 1's
                            self.recourse_experiment_params.n_oracle_positive[
                                repetition_idx] += oracle_value

                            # when oracle == 1, count 1's
                            self.recourse_experiment_params.n_shap_true_positive[
                                repetition_idx] += oracle_value * shap_value
                            self.recourse_experiment_params.n_lime_true_positive[
                                repetition_idx] += oracle_value * lime_value
                            self.recourse_experiment_params.n_intgradA_true_positive[
                                repetition_idx] += oracle_value * intgradA_value
                            self.recourse_experiment_params.n_intgradB_true_positive[
                                repetition_idx] += oracle_value * intgradB_value
                            self.recourse_experiment_params.n_smoothgrad_true_positive[
                                repetition_idx] += oracle_value * smoothgrad_value
                            self.recourse_experiment_params.n_grad_true_positive[
                                repetition_idx] += oracle_value * grad_value

                            # when oracle == 0, count 0's
                            self.recourse_experiment_params.n_shap_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - shap_value)
                            self.recourse_experiment_params.n_lime_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - lime_value)
                            self.recourse_experiment_params.n_intgradA_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - intgradA_value)
                            self.recourse_experiment_params.n_intgradB_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - intgradB_value)
                            self.recourse_experiment_params.n_smoothgrad_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - smoothgrad_value)
                            self.recourse_experiment_params.n_grad_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - grad_value)

                            self.recourse_experiment_params.n_experiment_samples[repetition_idx] += 1

                        elif experiment_type == experiment.ExperimentType.SPURIOUS:
                            shap_value = 1. * (np.abs(shap_values[feature]) > threshold)
                            lime_value = 1. * (np.abs(lime_values[feature]) > threshold)
                            intgradA_value = 1. * (np.abs(intgradA_values[feature]) > threshold)
                            intgradB_value = 1. * (np.abs(intgradB_values[feature]) > threshold)
                            smoothgrad_value = 1. * (np.abs(smoothgrad_values[feature]) > threshold)
                            grad_value = 1. * (np.abs(grad_values[feature]) > threshold)

                            """Bad style to have to copy this code."""
                            # how many 1's
                            self.spurious_experiment_params.n_oracle_positive[
                                repetition_idx] += oracle_value

                            # when oracle == 1, count 1's
                            self.spurious_experiment_params.n_shap_true_positive[
                                repetition_idx] += oracle_value * shap_value
                            self.spurious_experiment_params.n_lime_true_positive[
                                repetition_idx] += oracle_value * lime_value
                            self.spurious_experiment_params.n_intgradA_true_positive[
                                repetition_idx] += oracle_value * intgradA_value
                            self.spurious_experiment_params.n_intgradB_true_positive[
                                repetition_idx] += oracle_value * intgradB_value
                            self.spurious_experiment_params.n_smoothgrad_true_positive[
                                repetition_idx] += oracle_value * smoothgrad_value
                            self.spurious_experiment_params.n_grad_true_positive[
                                repetition_idx] += oracle_value * grad_value

                            # when oracle == 0, count 0's
                            self.spurious_experiment_params.n_shap_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - shap_value)
                            self.spurious_experiment_params.n_lime_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - lime_value)
                            self.spurious_experiment_params.n_intgradA_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - intgradA_value)
                            self.spurious_experiment_params.n_intgradB_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - intgradB_value)
                            self.spurious_experiment_params.n_smoothgrad_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - smoothgrad_value)
                            self.spurious_experiment_params.n_grad_true_negative[
                                repetition_idx] += (1 - oracle_value) * (1 - grad_value)

                            self.spurious_experiment_params.n_experiment_samples[repetition_idx] += 1

                        else:
                            raise ValueError('Unknown ExperimentType.')

        if experiment_type == experiment.ExperimentType.RECOURSE:
            self.recourse_experiment_params.save(data_basepath + self.experiment_name + '/' + experiment.ExperimentType.RECOURSE.value)
        elif experiment_type == experiment.ExperimentType.SPURIOUS:
            self.spurious_experiment_params.save(data_basepath + self.experiment_name + '/' + experiment.ExperimentType.SPURIOUS.value)
        else:
            raise ValueError('Unknown ExperimentType.')
