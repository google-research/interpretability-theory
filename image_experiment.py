"""Instantiate experiment object for imgae data.

This file runs the experiment with code designed specifically for image data;
i.e., the feature attribution is performed on pixels.
"""

import enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import experiment
import interpretability_methods
import datasets
import models

model_basepath = 'saved-data/models/'
data_basepath = 'saved-data/experiments/'


class ImageExperiment(experiment.Experiment):
    """Object for experiments with image data."""
    features_data: np.ndarray
    labels_data: np.ndarray
    standardize_data: Callable[[np.ndarray], np.ndarray]
    tf_standardize_data: Any
    train_generator: Any
    test_generator: Any
    data_info: Any

    def __init__(self, data: datasets.ImageDataset) -> None:
        super().__init__(data.value)
        self.features_data, self.labels_data, self.standardize_data, self.tf_standardize_data, self.train_generator, self.test_generator, self.data_info = datasets.import_image_dataset(data)
        self.model_architecture = models.create_image_model(data)

    def train_models(self, n_models: int, epochs: int, batch_size: int) -> None:
        """Train image models where dataset needs batches."""

        train_generator = self.train_generator.map(self.tf_standardize_data, num_parallel_calls=tf.data.AUTOTUNE)
        train_generator = train_generator.cache()
        train_generator = train_generator.shuffle(self.data_info.splits['train'].num_examples)
        train_generator = train_generator.batch(batch_size)
        train_generator = train_generator.prefetch(tf.data.AUTOTUNE)

        test_generator = self.test_generator.map(self.tf_standardize_data, num_parallel_calls=tf.data.AUTOTUNE)
        test_generator = test_generator.batch(batch_size)
        test_generator = test_generator.cache()
        test_generator = test_generator.prefetch(tf.data.AUTOTUNE)

        for model_idx in range(len(self.trained_models), len(self.trained_models) + n_models):
            model = self.model_architecture()
            model.summary(print_fn=print)

            history = model.fit(
                train_generator,
                epochs=epochs,
                verbose=2,
                validation_data=test_generator)

            history_df = pd.DataFrame(history.history)
            with open(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx) + '_history.csv', mode='w') as f:
                history_df.to_csv(f)
            self.training_history.append(history_df)

            model.save_weights(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx))
            self.trained_models.append(model)

    def perturbation(self, example: np.ndarray, x_axis: int, y_axis: int) -> np.ndarray:
        """Perturbations are always done on the actual scale, and then standardized to match the NN."""

        # hardcoded constants that will change interpretation
        n_steps = 20 # this should be even
        percent_perturb = 0.1

        value_range = np.max(self.features_data[:,x_axis,y_axis,:]) - np.min(self.features_data[:,x_axis,y_axis,:])
        perturbations = np.linspace(start=-percent_perturb*value_range, stop=percent_perturb*value_range, num=n_steps)

        perturbed_example = np.array([example for _ in range(n_steps)])
        perturbed_example[:,x_axis,y_axis,:] = np.array([example[x_axis,y_axis,:] + perturbations[i] for i in range(n_steps)])

        return perturbed_example

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """ Used to normalize the outputs of interpretability methods."""
        min_val = np.min(values)
        max_val = np.max(values)
        normal_vals = 2*(values - min_val) / np.max(np.array([(max_val-min_val),1e-10])) - 1
        return np.mean(normal_vals, axis=2)


    def run_experiment(self, experiment_type: experiment.ExperimentType, n_thresholds: int = 1, n_examples: int = 10, n_pixels: int = 10) -> None:
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

        # each model uses the same examples and pixels
        examples = self.features_data[np.random.randint(0, self.features_data.shape[0], size=n_examples)]
        n_spurious_examples = 100
        spurious_examples =  self.features_data[np.random.randint(0, self.features_data.shape[0], size=n_spurious_examples)]
        pixels = [[np.random.randint(0,self.features_data.shape[1],1), np.random.randint(0,self.features_data.shape[2],1)] for _ in range(n_pixels)]

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
                    for spurious_pixel in pixels:
                        perturbed_spurious_example = self.perturbation(spurious_example, spurious_pixel[0], spurious_pixel[1])
                        perturbed_output = trained_model_evaluated(self.standardize_data(np.array(perturbed_spurious_example)))[:,spurious_class]
                        spurious_variances.append(np.var(perturbed_output))

                spurious_quantile = np.quantile(spurious_variances, 0.8)

            for example_idx in range(n_examples):

                # new images for each model
                example = examples[example_idx]
                max_class = np.argmax(trained_model(self.standardize_data(example.reshape((1,)+example.shape))))

                shap_values = interpretability_methods.image_shap(
                        trained_model_evaluated,
                        self.standardize_data(example),
                        max_class)
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

                # use the selected pixels
                for pixel in pixels:

                    perturbed_example = self.perturbation(example, pixel[0], pixel[1])
                    perturbed_output = trained_model_evaluated(self.standardize_data(np.array(perturbed_example)))[:,max_class]

                    # unchanged from tabular
                    if experiment_type == experiment.ExperimentType.RECOURSE:
                        oracle_value = 1. * (np.mean(perturbed_output[int(len(perturbed_output)/2):len(perturbed_output)]) > np.mean(perturbed_output[0:int(len(perturbed_output)/2)]))
                    elif experiment_type == experiment.ExperimentType.SPURIOUS:
                        oracle_value = 1. * (np.var(perturbed_output) > spurious_quantile)
                    else:
                        raise ValueError('Unknown ExperimentType.')

                    for threshold_idx, threshold in enumerate(threshold_vals):

                        repetition_idx = trained_model_idx * len(threshold_vals) + threshold_idx

                        if experiment_type == experiment.ExperimentType.RECOURSE:
                            shap_value = 1. * (shap_values[pixel[0], pixel[1]] > threshold)
                            lime_value = 1. * (lime_values[pixel[0], pixel[1]] > threshold)
                            intgradA_value = 1. * (intgradA_values[pixel[0], pixel[1]] > threshold)
                            intgradB_value = 1. * (intgradB_values[pixel[0], pixel[1]] > threshold)
                            smoothgrad_value = 1. * (smoothgrad_values[pixel[0], pixel[1]] > threshold)
                            grad_value = 1. * (grad_values[pixel[0], pixel[1]] > threshold)

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
                            shap_value = 1. * (np.abs(shap_values[pixel[0], pixel[1]]) > threshold)
                            lime_value = 1. * (np.abs(lime_values[pixel[0], pixel[1]]) > threshold)
                            intgradA_value = 1. * (np.abs(intgradA_values[pixel[0], pixel[1]]) > threshold)
                            intgradB_value = 1. * (np.abs(intgradB_values[pixel[0], pixel[1]]) > threshold)
                            smoothgrad_value = 1. * (np.abs(smoothgrad_values[pixel[0], pixel[1]]) > threshold)
                            grad_value = 1. * (np.abs(grad_values[pixel[0], pixel[1]]) > threshold)

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
