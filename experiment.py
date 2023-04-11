"""Experiments for interpretability theory.

These experiments evaluate the ability of common interpretability methods
to infer the behaviour of real models on real data.

This class is used to store the base information needed to understand the
experiment. It will be inherited by downstream classes that instantiate it
for tabular and image data.

Visualizing a hypothesis test should be independent of what the test was
and how it was conducted -- all that is needed is the outputs of sensitivity,
specificity, and accuracy.

"""

import dataclasses
from dataclass_csv import DataclassWriter
from dacite import from_dict
import csv
from ast import literal_eval
from typing import Optional, Callable, Any
from sklearn.model_selection import train_test_split
import enum
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

model_basepath = 'saved-data/models/'
experiment_basepath = 'saved-data/experiments/'
img_basepath = 'saved-data/plots/'


class ExperimentType(enum.Enum):
    RECOURSE = 'recourse'
    SPURIOUS = 'spurious'


@dataclasses.dataclass
class ExperimentParams:
    """Keeps track of metadata from experiments.

    Each entry is a piece of metadata which is an array, where each entry
    of the array corresponds to the data from a single experiment.
    """
    n_shap_true_negative: np.ndarray = np.array([])
    n_lime_true_negative: np.ndarray = np.array([])
    n_intgradA_true_negative: np.ndarray = np.array([])
    n_intgradB_true_negative: np.ndarray = np.array([])
    n_smoothgrad_true_negative: np.ndarray = np.array([])
    n_grad_true_negative: np.ndarray = np.array([])
    n_shap_true_positive: np.ndarray = np.array([])
    n_lime_true_positive: np.ndarray = np.array([])
    n_intgradA_true_positive: np.ndarray = np.array([])
    n_intgradB_true_positive: np.ndarray = np.array([])
    n_smoothgrad_true_positive: np.ndarray = np.array([])
    n_grad_true_positive: np.ndarray = np.array([])
    n_oracle_positive: np.ndarray = np.array([])
    n_experiment_samples: np.ndarray = np.array([])

    def reset(self, n_repetitions: int) -> None:
        # reset global variables to store experiment metadata
        for field in self.__dataclass_fields__:
            setattr(self, field, np.zeros(n_repetitions))

    def save(self, filepath: str) -> None:
        # save experiment params to csv
        with open(filepath + '.csv', 'w') as f:
            w = DataclassWriter(f,
                                [self],
                                ExperimentParams)
            w.write()


class Experiment():
    """Object to run interpretability experiments."""

    image_dim: int
    model_architecture: Callable[[], tf.keras.models.Sequential]
    trained_models: list[tf.keras.models.Sequential] = []
    training_history: list[pd.DataFrame] = []
    experiment_name: str = ''
    experiment_params: ExperimentParams

    def __init__(self, name: str) -> None:
        self.recourse_experiment_params = ExperimentParams()
        self.spurious_experiment_params = ExperimentParams()
        self.experiment_name = name

    def load_models(self, n_models: int) -> None:
        self.trained_models = []
        for model_idx in range(n_models):

            history_df = pd.read_csv(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx) + '_history.csv')
            self.training_history.append(history_df)

            model = self.model_architecture()
            model.load_weights(model_basepath + self.experiment_name + '/' + self.experiment_name + '_' + str(model_idx)).expect_partial()
            self.trained_models.append(model)

    def load_experiment(self, experiment_type: ExperimentType) -> None:
        '''
        When run_experiment is called, the experiment_params are automatically
        saved to a csv file. This method reads such a csv into a
        dictionary, processes the elements of the dictionary, and then converts
        the dictionary back into a dataclass. The new experiment_params are
        set to be equal to this dataclass object.
        '''
        with open(experiment_basepath + self.experiment_name + '/' + experiment_type.value + '.csv') as f:
            reader = csv.DictReader(f)
            params_dict = next(reader)

        params_dict_spec = dict([key, np.array(literal_eval(val.replace('[ ', '[').replace(' ]', ']').replace('   ', ' ').replace('  ', ' ').replace('\n', '').replace(' ', ',')))] for key, val in params_dict.items())
        self.recourse_experiment_params = from_dict(data_class=ExperimentParams, data=params_dict_spec)

    def _sensitivity(self, n_true_positive: np.ndarray, experiment_type: ExperimentType) -> np.ndarray:
        """Computes the sensitivity (a probability).

        Args:
        n_true_positive: each element of the array is the number of times a
        positive oracle was correctly identified for that experiment

        Returns:
        each element of the array is the number of true positives divided
        by total number of positives for that experiment
        """
        if experiment_type == ExperimentType.RECOURSE:
            experiment_params = self.recourse_experiment_params
        elif experiment_type == ExperimentType.SPURIOUS:
            experiment_params = self.spurious_experiment_params
        else:
            raise ValueError('Unknown ExperimentType.')

        return np.array([
            round(x, 3)
            for x in n_true_positive / experiment_params.n_oracle_positive
        ])

        #np.array([round(x, 3) for x in wine_experiment.recourse_experiment_params.n_shap_true_positive / wine_experiment.recourse_experiment_params.n_oracle_positive])

    def _specificity(self, n_true_negative: np.ndarray, experiment_type: ExperimentType) -> np.ndarray:
        """Computes the sprecificity (a probability).

        Args:
        n_true_negative: each element of the array is the number of times a
        negative oracle was correctly identified for that experiment

        Returns:
        each element of the array is the number of true negatives divided
        by total number of negatives for that experiment
        """
        if experiment_type == ExperimentType.RECOURSE:
            experiment_params = self.recourse_experiment_params
        elif experiment_type == ExperimentType.SPURIOUS:
            experiment_params = self.spurious_experiment_params
        else:
            raise ValueError('Unknown ExperimentType.')

        return np.array([
            round(x, 3) for x in n_true_negative /
            (experiment_params.n_experiment_samples - experiment_params.n_oracle_positive)
        ])

    def _accuracy(self, n_true_positive: np.ndarray,
                  n_true_negative: np.ndarray,
                  experiment_type: ExperimentType) -> np.ndarray:
        """Computes the accuracy (a probability) for each experiment.

        Args:
        n_true_positive: each element of the array is the number of times a
        positive oracle was correctly identified for that experiment
        n_true_negative: each element of the array is the number of times a
        negative oracle was correctly identified for that experiment

        Returns:
        each element of the array is the number of true negatives or true
        positives divided by total number of
        observations for that experiment
        """
        if experiment_type == ExperimentType.RECOURSE:
            experiment_params = self.recourse_experiment_params
        elif experiment_type == ExperimentType.SPURIOUS:
            experiment_params = self.spurious_experiment_params
        else:
            raise ValueError('Unknown ExperimentType.')

        return np.array([
            round(x, 3) for x in (n_true_positive + n_true_negative) /
            experiment_params.n_experiment_samples
        ])

    def visualize_hypothesis_test(self,
                                  experiment_type: ExperimentType) -> None:
        """Plot the specificity and sensitivity relative to the trivial
        baseline.
        """
        if experiment_type == ExperimentType.RECOURSE:
            experiment_params = self.recourse_experiment_params
        elif experiment_type == ExperimentType.SPURIOUS:
            experiment_params = self.spurious_experiment_params
        else:
            raise ValueError('Unknown ExperimentType.')

        ax = plt.subplots()[1]
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        xval = np.linspace(0, 1, 20)
        yval = np.linspace(0, 1, 20)
        ax.plot(xval, yval, linestyle='dashed', color='grey')
        ax.grid(alpha=0.75)

        title_str = self.experiment_name + ' ' + experiment_type.value
        # ax.set(xlabel='Pr(False Positive)', ylabel='Pr(True Positive)',
        #        title=string.capwords(title_str))
        ax.set(xlabel='Pr(False Positive)', ylabel='Pr(True Positive)',
               title=title_str)

        sensitivities = np.array([
            self._sensitivity(experiment_params.n_shap_true_positive, experiment_type),
            self._sensitivity(experiment_params.n_intgradA_true_positive, experiment_type),
            self._sensitivity(experiment_params.n_intgradB_true_positive, experiment_type),
            self._sensitivity(experiment_params.n_smoothgrad_true_positive, experiment_type),
            self._sensitivity(experiment_params.n_lime_true_positive, experiment_type),
            self._sensitivity(experiment_params.n_grad_true_positive, experiment_type)
        ])
        specificities = np.array([
            self._specificity(experiment_params.n_shap_true_negative, experiment_type),
            self._specificity(experiment_params.n_intgradA_true_negative, experiment_type),
            self._specificity(experiment_params.n_intgradB_true_negative, experiment_type),
            self._specificity(experiment_params.n_smoothgrad_true_negative, experiment_type),
            self._specificity(experiment_params.n_lime_true_negative, experiment_type),
            self._specificity(experiment_params.n_grad_true_negative, experiment_type)
        ])

        colors = ['red', 'lightskyblue', 'dodgerblue', 'darkviolet', 'limegreen', 'gold']

        for i in range(len(sensitivities)):
          ax.scatter(1-specificities[i], sensitivities[i], c=colors[i], s=5.0)
        ax.scatter(0., 1., color='black')

        ax.legend(['Trivial', 'SHAP', 'IG(Mean)', 'IG(Min)', 'SG', 'LIME', 'GRAD', 'Optimal'])

        with open(img_basepath + self.experiment_name + '-' + experiment_type.value +
                '-hypothesis-tradeoff' + '.png', 'wb') as f:
            plt.savefig(f, format='png')
