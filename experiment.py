"""Experiments for interpretability theory.

These experiments evaluate the ability of common interpretability methods
to infer the behaviour of real models on real data.

This class is used to store the base information needed to understand the
experiment. It will be inherited by downstream classes that instantiate it
for tabular and image data.

Visualizing a hypothesis test should be independent of what the test was and how
it was conducted -- all that is needed is the outputs of sensitivity,
specificity, and accuracy.

Currently, visualization is done in-line in a colab by default, but will save to
a filename if passed one.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class Experiment():
  """Object to run interpretability experiments."""

  def __init__(self) -> None:
    self.features_data = None
    self.labels_data = None
    self.n_features = None
    self.n_classes = None
    self.n_data = None
    self.feature_names = None

    # output 0 when oracle is 0 (specificity, or true negative)
    self.n_shap_true_negative = None
    self.n_lime_true_negative = None
    self.n_intgrad_true_negative = None
    self.n_grad_true_negative = None
    # output 1 when oracle is 1 (sensitivity, or true positive)
    self.n_shap_true_positive = None
    self.n_lime_true_positive = None
    self.n_intgrad_true_positive = None
    self.n_grad_true_positive = None
    # how often oracle is 1
    self.n_oracle_positive = None

    # counters for experiments
    self.n_experiment_samples = None
    self.model_accuracy = None

  def _sensitivity(self, n_true_positive: np.ndarray) -> np.ndarray:
    """Computes the sensitivity (a probability).

    Args:
      n_true_positive: number of times a positive oracle was correctly
        identified

    Returns:
      number of true positives divided by total number of positives
    """
    return np.array(
        list(
            map(lambda x: round(x, 3),
                n_true_positive / self.n_oracle_positive)))

  def _specificity(self, n_true_negative: np.ndarray) -> np.ndarray:
    """Computes the sprecificity (a probability).

    Args:
      n_true_negative: number of times a negative oracle was correctly
        identified

    Returns:
      number of true negatives divided by total number of negatives
    """
    return np.array(
        list(
            map(
                lambda x: round(x, 3), n_true_negative /
                (self.n_experiment_samples - self.n_oracle_positive))))

  def _accuracy(self, n_true_positive: np.ndarray,
                n_true_negative: np.ndarray) -> np.ndarray:
    """Computes the accuracy (a probability).

    Args:
      n_true_positive: number of times a positive oracle was correctly
        identified
      n_true_negative: number of times a negative oracle was correctly
        identified

    Returns:
      number of true negatives or true positives divided by total number of
      observations
    """
    return np.array(
        list(
            map(lambda x: round(x, 3), (n_true_positive + n_true_negative) /
                self.n_experiment_samples)))

  def visualize_experiment_table(self) -> None:
    """Print out the specificity, sensitivity, and accuracy of the test."""
    print('Pr(shap(f,x)=1 | oracle(f,x)=1) = {x}'.format(
        x=self._sensitivity(self.n_shap_true_positive)))
    print('Pr(lime(f,x)=1 | oracle(f,x)=1) = {x}'.format(
        x=self._sensitivity(self.n_lime_true_positive)))
    print('Pr(IG(f,x)=1 | oracle(f,x)=1) = {x}'.format(
        x=self._sensitivity(self.n_intgrad_true_positive)))
    print('Pr(grad(f,x)=1 | oracle(f,x)=1) = {x}'.format(
        x=self._sensitivity(self.n_grad_true_positive)))
    print('Pr(shap(f,x)=0 | oracle(f,x)=0) = {x}'.format(
        x=self._specificity(self.n_shap_true_negative)))
    print('Pr(lime(f,x)=0 | oracle(f,x)=0) = {x}'.format(
        x=self._specificity(self.n_lime_true_negative)))
    print('Pr(IG(f,x)=0 | oracle(f,x)=0) = {x}'.format(
        x=self._specificity(self.n_intgrad_true_negative)))
    print('Pr(grad(f,x)=0 | oracle(f,x)=0) = {x}'.format(
        x=self._specificity(self.n_grad_true_negative)))
    print('Pr(shap(f,x) matches oracle(f,x)) = {x}'.format(
        x=self._accuracy(self.n_shap_true_positive, self.n_shap_true_negative)))
    print('Pr(lime(f,x) matches oracle(f,x)) = {x}'.format(
        x=self._accuracy(self.n_lime_true_positive, self.n_lime_true_negative)))
    print('Pr(IG(f,x) matches oracle(f,x)) = {x}'.format(
        x=self._accuracy(self.n_intgrad_true_positive,
                         self.n_intgrad_true_negative)))
    print('Pr(grad(f,x) matches oracle(f,x)) = {x}'.format(
        x=self._accuracy(self.n_grad_true_positive, self.n_grad_true_negative)))
    print('Pr(oracle(f,x)=1) = {x}'.format(
        x=np.array(
            list(
                map(lambda x: round(x, 3), self.n_oracle_positive /
                    self.n_experiment_samples)))))

  def visualize_hypothesis_test(self,
                                title: str,
                                filepath: Optional[str] = None) -> None:
    """Plot the specificity and sensitivity relative to the trivial baseline."""

    ax = plt.subplots()[1]
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])

    xval = np.linspace(0, 1, 20)
    yval = np.linspace(1, 0, 20)
    ax.plot(xval, yval, linestyle='dashed', color='grey')
    ax.grid(alpha=0.75)

    ax.set(xlabel='Pr(True Positive)', ylabel='Pr(True Negative)', title=title)

    sensitivities = np.array([
        self._sensitivity(self.n_shap_true_positive),
        self._sensitivity(self.n_intgrad_true_positive),
        self._sensitivity(self.n_lime_true_positive),
        self._sensitivity(self.n_grad_true_positive)
    ])
    specificities = np.array([
        self._specificity(self.n_shap_true_negative),
        self._specificity(self.n_intgrad_true_negative),
        self._specificity(self.n_lime_true_negative),
        self._specificity(self.n_grad_true_negative)
    ])

    colors = ['blue', 'orange', 'green', 'red']

    for i in range(len(sensitivities)):
      ax.scatter(sensitivities[i], specificities[i], color=colors[i])
    ax.scatter(1., 1., color='black')

    ax.legend(['Trivial', 'SHAP', 'IG', 'LIME', 'GRAD', 'Optimal'])

    if filepath is not None:
      with gfile.Open(filepath + '.png', 'wb') as f:
        plt.savefig(f, format='png')

  def visualize_accuracy(self,
                         title: str,
                         filepath: Optional[str] = None) -> None:
    """Plot the accuracy relative to the trivial baseline."""

    ax = plt.subplots()[1]
    ax.set_ylim([-0.05, 1.05])

    n_repetitions = len(self.n_experiment_samples)

    accuracies = np.array([
        self._accuracy(self.n_shap_true_positive, self.n_shap_true_negative),
        self._accuracy(self.n_intgrad_true_positive,
                       self.n_intgrad_true_negative),
        self._accuracy(self.n_lime_true_positive, self.n_lime_true_negative),
        self._accuracy(self.n_grad_true_positive, self.n_grad_true_negative)
    ])
    mean_accuracies = np.mean(accuracies, axis=1)
    std_accuracies = 1.96 * np.std(accuracies, axis=1) / np.sqrt(n_repetitions)

    xlabels = ['SHAP', 'IG', 'LIME', 'GRAD']
    xvals = range(1, len(xlabels) + 1)
    colors = ['blue', 'orange', 'green', 'red']
    plt.bar(
        xvals, mean_accuracies, color=colors, alpha=0.5, yerr=std_accuracies)
    plt.xticks(xvals, xlabels)
    ax.set_xlim([np.min(xvals) - 0.55, np.max(xvals) + 0.55])

    trivial_accuracy = np.max(
        (self.n_oracle_positive / self.n_experiment_samples, 1 -
         (self.n_oracle_positive / self.n_experiment_samples)),
        axis=0)
    mean_trivial_accuracy = np.mean(trivial_accuracy)
    std_trivial_accuracy = 1.96 * np.std(trivial_accuracy) / np.sqrt(
        n_repetitions)
    plt.hlines(
        y=mean_trivial_accuracy,
        xmin=np.min(xvals) - 0.55,
        xmax=np.max(xvals) + 0.55,
        linestyle='dashed',
        color='grey')
    plt.vlines(
        x=np.mean(xvals),
        ymin=mean_trivial_accuracy - std_trivial_accuracy,
        ymax=mean_trivial_accuracy + std_trivial_accuracy,
        color='grey')
    plt.hlines(
        y=1.0,
        xmin=np.min(xvals) - 0.55,
        xmax=np.max(xvals) + 0.55,
        linestyle='dashed',
        color='black')
    plt.text(0.5, mean_trivial_accuracy + 0.02, 'Trivial')

    ax.set(xlabel='', ylabel='Accuracy', title=title)

    if filepath is not None:
      with open(filepath + '.png', 'wb') as f:
        plt.savefig(f, format='png')
