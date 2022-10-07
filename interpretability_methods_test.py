"""Tests for experiment functions.

Specifically, whether the implementations of Shap, IG, Grad, and Lime work as
expected for 3 functions that are easy to compute exactly.
"""

import numpy as np
import tensorflow as tf

import interpretability_methods


def test_linear_function(self) -> None:

  def model(x: np.ndarray) -> np.ndarray:
    return x @ np.array([[2], [-1]])

  def feature_sampler(n: int) -> np.ndarray:
    return np.random.rand(n, 2)

  example = np.array([0.4, 0.6])

  shap_exact = np.array([[2 * example[0] - 1], [-example[1] + 0.5]])
  intgrad_exact = np.array([[2 * example[0]], [-example[1]]])
  grad_exact = np.array([[2], [-1]])
  lime_unregularized_exact = np.array([[2], [-1]])

  shap_value = interpretability_methods.shap(
      model,
      example, [0],
      np.array([0, 1]),
      feature_sampler(10000),
      num_iters=10000)
  intgrad_value = interpretability_methods.integrated_gradient(
      model, example, [0], np.array([0, 1]))
  grad_value = interpretability_methods.gradient(model, example, [0],
                                                 np.array([0, 1]))
  lime_unregularized_value = interpretability_methods.lime(
      model, example, [0], np.array([0, 1]), reg_param=0., num_iters=10000)
  lime_regularized_value = interpretability_methods.lime(
      model, example, [0], np.array([0, 1]), reg_param=2., num_iters=10000)

  assertAlmostEqual(
      np.sum(np.abs(shap_value - shap_exact)),
      0,
      delta=0.1,
      msg='Shap failed for linear function')
  assertAlmostEqual(
      np.sum(np.abs(intgrad_value - intgrad_exact)),
      0,
      delta=0.1,
      msg='IG failed for linear function')
  assertAlmostEqual(
      np.sum(np.abs(grad_value - grad_exact)),
      0,
      delta=0.1,
      msg='Grad failed for linear function')

  # LIME test
  # hard to solve for regularized least squares exactly
  # instead check the answer is correct for just least squares
  # and that the regularized answer has smaller L2 norm
  assertAlmostEqual(
      np.sum(np.abs(lime_unregularized_value - lime_unregularized_exact)),
      0,
      delta=0.1,
      msg='Lime failed for linear function -- incorrect solution')
  assertLess(
      np.sum(np.square(lime_regularized_value)),
      1.1 * np.sum(np.square(lime_unregularized_value)),
      msg='Lime failed for linear function -- regularization is not smaller')

def test_nonlinear_function(self) -> None:

  def model(x: np.ndarray) -> np.ndarray:
    return x**2 * np.array([1., 2.])

  def feature_sampler(n: int) -> np.ndarray:
    return np.random.rand(n, 1)

  example = np.array([3.])

  shap_exact = np.array(
      [[example[0]**2 - (1 / 3), 2 * example[0]**2 - (2 / 3)]])
  intgrad_exact = np.array([[example[0]**2, 2 * example[0]**2]])
  grad_exact = np.array([[2 * example[0], 4 * example[0]]])
  lime_unregularized_exact = np.array([[
      (3 * example[0] *
       (0.1)**2 + example[0]**3) / ((0.1)**2 + example[0]**2), 2 *
      (3 * example[0] * (0.1)**2 + example[0]**3) / ((0.1)**2 + example[0]**2)
  ]])

  shap_value = interpretability_methods.shap(
      model,
      example, [0, 1],
      np.array([0]),
      feature_sampler(10000),
      num_iters=10000)
  intgrad_value = interpretability_methods.integrated_gradient(
      model, example, [0, 1], np.array([0]))
  grad_value = interpretability_methods.gradient(model, example, [0, 1],
                                                 np.array([0]))
  lime_unregularized_value = interpretability_methods.lime(
      model,
      example, [0, 1],
      np.array([0]),
      local_radius=0.1,
      reg_param=0.,
      num_iters=10000)
  lime_regularized_value = interpretability_methods.lime(
      model,
      example, [0, 1],
      np.array([0]),
      local_radius=0.1,
      reg_param=2.,
      num_iters=10000)

  assertAlmostEqual(
      np.sum(np.abs(shap_value - shap_exact)),
      0,
      delta=0.1,
      msg='Shap failed for nonlinear function')
  assertAlmostEqual(
      np.sum(np.abs(intgrad_value - intgrad_exact)),
      0,
      delta=0.1,
      msg='IG failed for nonlinear function')
  assertAlmostEqual(
      np.sum(np.abs(grad_value - grad_exact)),
      0,
      delta=0.1,
      msg='Grad failed for nonlinear function')

  # LIME test
  # hard to solve for regularized least squares exactly
  # instead check the answer is correct for just least squares
  # and that the regularized answer has smaller L2 norm
  # (adjusted for random error)
  assertAlmostEqual(
      np.sum(np.abs(lime_unregularized_value - lime_unregularized_exact)),
      0,
      delta=0.1,
      msg='Lime failed for nonlinear function -- incorrect solution')
  assertLess(
      np.sum(np.square(lime_regularized_value)),
      1.1 * np.sum(np.square(lime_unregularized_value)),
      msg='Lime failed for nonlinear function -- regularization is not smaller'
  )

def test_feature_interaction(self) -> None:

  def model(x: np.ndarray) -> tf.Tensor:
    x = x * np.array([[1., 1.]])
    return tf.transpose(tf.maximum(x[:, 0], x[:, 1]) * np.array([[1.]]))

  def feature_sampler(n: int) -> np.ndarray:
    return np.random.rand(n, 2)

  example = np.array([1., 0.])

  shap_exact = np.array([[
      0.5 * (np.max(example) - 0.5 *
             (1 + example[1]**2) + example[0]**2 - 2 / 3)
  ],
                         [
                             0.5 *
                             (np.max(example) + 0.5 *
                              (1 + example[1]**2) - example[0]**2 - 2 / 3)
                         ]])
  intgrad_exact = np.array([[1.], [0.]])
  grad_exact = np.array([[1.], [0.]])

  shap_value = interpretability_methods.shap(
      lambda x: model(x).numpy(),
      example, [0],
      np.array([0, 1]),
      feature_sampler(10000),
      num_iters=10000)
  intgrad_value = interpretability_methods.integrated_gradient(
      model, example, [0], np.array([0, 1]))
  grad_value = interpretability_methods.gradient(model, example, [0],
                                                 np.array([0, 1]))

  assertAlmostEqual(
      np.sum(np.abs(shap_value - shap_exact)),
      0,
      delta=0.1,
      msg='Shap failed for feature interaction')
  assertAlmostEqual(
      np.sum(np.abs(intgrad_value - intgrad_exact)),
      0,
      delta=0.1,
      msg='IG failed for feature interaction')
  assertAlmostEqual(
      np.sum(np.abs(grad_value - grad_exact)),
      0,
      delta=0.1,
      msg='Grad failed for feature interaction')

def test_lime_feature_interaction(self) -> None:
  """Test feature interaction for Lime.

  Computing Lime by hand is a pain for the function max(x,y).
  Instead, compute it by hand for x*y.
  Still a pain to solve for the exact solution, so just check that both score
  equations are satisfied.
  """

  def model(x: np.ndarray) -> np.ndarray:
    return (np.diag(x @ np.array([[0, 1], [0, 0]]) @ x.T * np.array([[1.]])) *
            np.array([[1.]])).T

  example = np.array([2, 3])
  print(model(example))

  lime_unregularized_value = interpretability_methods.lime(
      model, example, [0], np.array([0, 1]), reg_param=0, num_iters=10000)

  assertAlmostEqual(
      np.max([
          np.abs(lime_unregularized_value[0, 0] - example[1] +
                 lime_unregularized_value[1, 0] * example[0] * example[1] /
                 ((0.1)**2 + example[0]**2)),
          np.abs(lime_unregularized_value[1, 0] - example[0] +
                 lime_unregularized_value[0, 0] * example[0] * example[1] /
                 ((0.1)**2 + example[1]**2))
      ]),
      0,
      delta=0.1,
      msg='Lime failed for feature interaction -- incorrect solution')


