# Code from causal_sensitivity_2020 repository by @zachwooddoughty

import itertools
import math
import logging
import os
import json
import argparse
from collections import OrderedDict, defaultdict
import random
from statistics import mean
from statistics import variance

import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from datasets import SyntheticData
from utils import Distribution, gformula, NumpySerializer
from utils import get_dist, get_fractional_dist
from utils import construct_proxy_dist
from sensitivity import clopper_pearson_interval

from line_profiler import LineProfiler

import matplotlib.pyplot as plt


def get_assn(truth_val, confound_assn, proxy_i):
  '''
    helper function to take in proxy variable value, confounding variable assn, and proxy index
    and return the distribution of the three
  '''
  retval = list(confound_assn)
  retval.insert(proxy_i, truth_val)
  return tuple(retval)


def get_error_rate(proxy, truth, truth_val, sample=False,
                   alpha=None, confounds=None, confound_assn=None,
                   n_func=None, return_triple=False):
  '''
  Given a proxy (A*) and truth (A), calculate the error rate when A=1.
  If confounds and confound_assn are given, limit this calculation to when
    confounds have the given assignment
  '''

  assert type(proxy) == np.ndarray
  assert type(truth) == np.ndarray

  where = []
  if confounds is not None and confound_assn is not None:
    where = [confounds[:, i] == confound_assn[i] for i in range(len(confound_assn))]

  true_where = where + [truth == truth_val]
  true_where = np.all(np.stack(true_where, axis=1), axis=1)
  true = np.sum(true_where)

  correct_where = where + [proxy == truth_val, truth == truth_val]
  correct_where = np.all(np.stack(correct_where, axis=1), axis=1)
  correct = np.sum(correct_where)

  if true > 0:
    err_rate = 1 - (correct / true)
  else:
    err_rate = 0

  if n_func is not None and true > 0:
    true = n_func(true)

  if alpha is None:
    if sample:
      return (err_rate, true)
    else:
      return (err_rate, )

  else:
    interval = clopper_pearson_interval(
        true, err_rate, alpha, return_triple=return_triple)
    return tuple(sorted(interval))


def get_fractional_error_rate(proxy, truth, truth_val, sample=False,
                              alpha=None, confounds=None, confound_assn=None,
                              n_func=None, return_triple=False):
  '''
  Calculate a fractional error rate value when proxy is a binary logit
  If confounds and confound_assn are given, limit this calculation to when
    confounds have the given assignment
  '''
  if confounds is not None and confound_assn is not None:
    where = np.equal(confounds[:, 0], confound_assn[0])
    for i in range(1, len(confound_assn)):
      where = np.logical_and(where, np.equal(confounds[:, i], confound_assn[i]))
    # where = [confounds[:, i] == confound_assn[i] for i in range(len(confound_assn))]
  else:
    where = np.ones_like(proxy, dtype=np.bool)

  # true_where = where + [truth == truth_val]
  # true_where = np.all(np.stack(true_where, axis=1), axis=1)
  true_where = np.logical_and(where, np.equal(truth, truth_val))
  true = np.sum(true_where)

  if true > 0:
    diff = 1 - truth[true_where].astype(np.float64) - proxy[true_where]
    err = np.sum(np.absolute(diff))
    err_rate = err / true
  else:
    err_rate = 0

  if n_func is not None and true > 0:
    true = n_func(true)

  if alpha is None:
    if sample:
      return (err_rate, true)
    else:
      return (err_rate, )

  else:
    interval = clopper_pearson_interval(
        true, err_rate, alpha, return_triple=return_triple)
    return tuple(sorted(interval))


# def count_where_assn(truth, truth_val, confounds, confound_assn):
#   where = [confounds[i, :] == confound_assn[i] for i in range(len(confound_assn))]
#   true_where = where + [truth == truth_val]
#   true_where = np.all(np.stack(true_where, axis=0), axis=0)
#   true = np.sum(true_where)
#   return true

def count_where_assn(arr, columns, mapping):
  assert type(arr) == np.ndarray
  assert type(mapping) == dict and len(mapping) > 0

  where = []
  for var, val in mapping.items():
    where.append(arr[:, columns.index(var)] == val)
  true = np.sum(np.all(np.stack(where, axis=1), axis=1))
  return true


def calculate_error_matrix(experiment_data, proxy_arr, truth_arr, proxy_var, columns, sample=False,
                           alpha=None, nondiff=False, debug=False,
                           n_func=None, return_triple=False, influence_vars = None):
  '''
    Given the proxy (A*) and truth (A), calculate the correction matrix
    to adjust the causal effect calculations
  '''

  assert type(proxy_arr) == np.ndarray
  assert type(truth_arr) == np.ndarray

  full_dim = len(columns)
  proxy_i = columns.index(proxy_var)

  # proxy_arr = np.round(proxy_arr, 0).astype(np.int64)
  assert proxy_arr.shape[1] == full_dim
  proxy_col = proxy_arr[:, proxy_i]
  if truth_arr.shape[1] == full_dim:
    truth_arr = truth_arr[:, proxy_i]
  else:
    assert truth_arr.shape[1] == 1

  get_error_rate_func = get_fractional_error_rate
  is_binary = np.all((proxy_arr == 0) | (proxy_arr == 1))
  if is_binary:
    get_error_rate_func = get_error_rate

  errs = {}
  if influence_vars is None:
    nondiff_errs = {}
    for truth_val in [0, 1]:
      nondiff_errs[truth_val] = get_error_rate_func(
          proxy_col, truth_arr, truth_val, alpha=alpha,
          n_func=n_func, return_triple=return_triple, sample=sample)
    # print("est errs", nondiff_errs[0], nondiff_errs[1])
    for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
      truth_val = assn[proxy_i]
      errs[assn] = nondiff_errs[truth_val]
  else:
    indices = [i for i in range(full_dim) if columns[i] != proxy_var and columns[i] in influence_vars]
    confounds = proxy_arr[:, indices]
    for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
      truth_val = assn[proxy_i]
      proxy_val = 1 - truth_val
      confound_assn = tuple(assn[i] for i in indices)
      errs[assn] = get_error_rate_func(
          proxy_col, truth_arr, proxy_val, sample=sample,
          confounds=confounds, confound_assn=confound_assn,
          alpha=alpha, n_func=n_func, return_triple=return_triple)


  to_return = Distribution(data=errs, columns=columns, normalized=False)
  experiment_data.set_err_matrix(to_return)
  return to_return


def get_corrected_dist(dist, err_dist, proxy_var, truth=None, debug=False):
  '''
  Given a proxy distribution dist, error estimates, and a held-out truth,
    calculate the new dist that comes from using the error estimates to correct the proxy.
  dist: p(C, A*, Y)
  errs: p(A*, A)
  truth: p(C, A, Y)
  '''

  assert type(dist) == Distribution

  non_proxy_columns = [col for col in dist.columns if col != proxy_var]
  full_dim = len(dist.columns)
  corrected = {}
  for non_proxy_assn in itertools.product(*[range(2) for _ in range(full_dim - 1)]):
    non_proxy_assn = dict(zip(non_proxy_columns, non_proxy_assn))

    assn0 = {**non_proxy_assn, **{proxy_var: 0}}
    tuple_assn0 = tuple(assn0[col] for col in dist.columns)
    assn1 = {**non_proxy_assn, **{proxy_var: 1}}
    tuple_assn1 = tuple(assn1[col] for col in dist.columns)
    err1 = err_dist.get(**assn1)
    err0 = err_dist.get(**assn0)
    # err0 = errs[get_assn(0, confound_assn, proxied_index)]

    if err1 + err0 != 1.0:
      corrected1 = (1 - err1) * dist.get(**assn0) - err1 * dist.get(**assn1)
      corrected1 /= (1 - err1 - err0)
      corrected1 = max(corrected1, 1e-5)
      corrected[tuple_assn1] = corrected1
    else:
      corrected[tuple_assn1] = dist.get(**assn1)

    if err1 + err0 != 1.0:
      corrected0 = - err0 * dist.get(**assn0) + (1 - err0) * dist.get(**assn1)
      corrected0 /= (1 - err1 - err0)
      corrected0 = max(corrected0, 1e-5)
      corrected[tuple_assn0] = corrected0
    else:
      corrected[tuple_assn0] = dist.get(**assn1)

  # We may need to normalize if we had singularities and had to keep original dist 
  total_weight = np.sum(list(corrected.values()))
  if not np.isclose(total_weight, 1, atol=1e-1):
    total_weight = math.fsum(list(corrected.values()))
    for i in range(3):
      keys, vals = zip(*list(corrected.items()))
      new_vals = np.array(vals) / total_weight
      corrected = dict(zip(keys, new_vals))
      new_total_weight = math.fsum(list(corrected.values()))
      if np.isclose(new_total_weight, 1, atol=1e-1):
        break

  corrected = Distribution(data=corrected, columns=dist.columns,
                           normalized=True)
  return corrected

def train_error_model(experiment_data, proxy_arr, truth_arr, proxy_var, columns, influence_vars=None):
  '''
    Given the proxy (A*) and truth (A), and all other features that influence A* (assumed
    to be all other variables if influence_vars is None) fits a logistic regression to
    predict an error in A* given by p(A* != A | A, C, Y).
  '''

  assert type(proxy_arr) == np.ndarray
  assert type(truth_arr) == np.ndarray

  full_dim = len(columns)
  proxy_i = columns.index(proxy_var)

  assert proxy_arr.shape[1] == full_dim
  proxy_col = proxy_arr[:, proxy_i]
  if truth_arr.shape[1] == full_dim:
    truth_arr = truth_arr[:, proxy_i]
  else:
    assert truth_arr.shape[1] == 1

  bin_proxy_col = (proxy_col > 0.5).astype(int)
  label_list = [0 if bin_proxy_col[i] == truth_arr[i] else 1 for i in range(len(proxy_col))]
  # import pdb; pdb.set_trace()

  # if influence_vars == None:
  #   indices = [i for i in range(full_dim) if columns[i] != proxy_var]
  # else:
  if influence_vars is not None:
    indices = [i for i in range(full_dim) if (columns[i] in influence_vars and columns[i] != proxy_var)]
    influences = np.column_stack((proxy_arr[:, indices], truth_arr))
  else:
    influences = truth_arr.reshape(-1, 1)

  model = LogisticRegression()
  model.fit(influences, label_list)

  experiment_data.set_model_coef(model.coef_)
  experiment_data.set_model_int(model.intercept_)
  experiment_data.model = model
  # print(accuracy_score(model.predict(influences), label_list))
  # import pdb; pdb.set_trace()
  return model

def get_regression_corrected_dist(dist, proxy_var, model, influence_vars = None):
  '''
  Given a proxy distribution dist, error estimates, and a held-out truth,
    calculate the new dist that comes from using the error estimates to correct the proxy.
  dist: p(C, A*, Y)
  errs: p(A*, A)
  truth: p(C, A, Y)
  '''

  assert type(dist) == Distribution

  # if influence_vars != None:
  #   model = train_error_model(proxy_arr, truth_arr, proxy_var, columns, influence_vars)
  # else:
  #   model = train_error_model(proxy_arr, truth_arr, proxy_var, columns)

  non_proxy_columns = [col for col in dist.columns if col != proxy_var]

  full_dim = len(dist.columns)
  corrected = {}
  if influence_vars is not None:
    influencers = influence_vars[:]
    print(influence_vars)
    influencers.append(proxy_var)
    print(influence_vars)

  print(influencers)
  for non_proxy_assn in itertools.product(*[range(2) for _ in range(full_dim - 1)]):
    non_proxy_assn = dict(zip(non_proxy_columns, non_proxy_assn))

    assn0 = {**non_proxy_assn, **{proxy_var: 1}}
    tuple_assn0 = tuple(assn0[col] for col in dist.columns)
    assn1 = {**non_proxy_assn, **{proxy_var: 0}}
    tuple_assn1 = tuple(assn1[col] for col in dist.columns)
    if influence_vars is not None:
      model_input1 = tuple(assn1[col] for col in influencers)
      model_input0 = tuple(assn0[col] for col in influencers)
      input = np.array(model_input1).reshape(1, -1)
      shape = input.shape
      # import pdb; pdb.set_trace()
      # print(influence_vars)
      # print(assn1)
      err1 = model.predict_proba(np.array(model_input1).reshape(1, -1))[:, 1]
      err0 = model.predict_proba(np.array(model_input0).reshape(1, -1))[:, 1]
    else: 
      err1 = model.predict_proba(np.array(assn1[proxy_var]).reshape(-1, 1))[:, 1]
      err0 = model.predict_proba(np.array(assn0[proxy_var]).reshape(-1, 1))[:, 1]
    # err0 = errs[get_assn(0, confound_assn, proxied_index)]

    if err1 + err0 != 1.0:
      corrected1 = (1 - err1) * dist.get(**assn0) - err1 * dist.get(**assn1)
      corrected1 /= (1 - err1 - err0)
      corrected1 = max(corrected1, 1e-5)
      corrected[tuple_assn1] = corrected1
    else:
      corrected[tuple_assn1] = dist.get(**assn1)

    if err1 + err0 != 1.0:
      corrected0 = - err0 * dist.get(**assn0) + (1 - err0) * dist.get(**assn1)
      corrected0 /= (1 - err1 - err0)
      corrected0 = max(corrected0, 1e-5)
      corrected[tuple_assn0] = corrected0
    else:
      corrected[tuple_assn0] = dist.get(**assn1)

  # We may need to normalize if we had singularities and had to keep original dist 
  total_weight = np.sum(list(corrected.values()))
  if not np.isclose(total_weight, 1, atol=1e-1):
    total_weight = math.fsum(list(corrected.values()))
    for i in range(3):
      keys, vals = zip(*list(corrected.items()))
      new_vals = np.array(vals) / total_weight
      corrected = dict(zip(keys, new_vals))
      new_total_weight = math.fsum(list(corrected.values()))
      if np.isclose(new_total_weight, 1, atol=1e-1):
        break

  corrected = Distribution(data=corrected, columns=dist.columns,
                           normalized=True)
  return corrected

def correct_with_model(experiment_data, dist, proxy_var, proxy_arr, truth_arr, influence_vars = None):
  
  experiment_data.set_p_dot(dist.dict)

  model = train_error_model(experiment_data, proxy_arr, truth_arr, proxy_var, dist.columns, influence_vars)
  return get_regression_corrected_dist(dist, proxy_var, model, influence_vars = influence_vars)

def impute_and_correct_with_model(experiment_data, train, test, columns, proxy_var,
                                  train_percent=0.5, c_dim=1, u_dim=1,
                                  sample_err_rates=0, bootstrap=1,
                                  nondiff=False, alpha=None, debug=False, influence_vars = None):
  '''
  train: training data
  test: testing data
  num_train: how many examples of training data to use for training,
    (leaving the rest for development)
  proxy_i: what is the index of the proxied variable (e.g. 1)
  confound_i: what are the indices of the proxy's confounders (e.g. 0, 2)
  '''

  # features are everything but the proxy variable
  full_dim = len(columns)
  proxy_i = columns.index(proxy_var)

  feature_rows = tuple(i for i in range(train.shape[1]) if i != proxy_i)
  # train_features = np.concatenate((train[:, :proxy_i], train[:, 1 + proxy_i:]),
                                  # axis=1)

  num_train = int(train_percent * train.shape[0])
  num_dev = train.shape[0] - num_train

  model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
  # model.fit(train_features[:num_train, :], train[:num_train, proxy_i])
  model.fit(train[:num_train, feature_rows], train[:num_train, proxy_i])

  # print(accuracy_score(model.predict(train[:num_train, feature_rows]), train[:num_train, proxy_i]))


  # true_dev = train_features[num_train:, :]
  # dev_preds = model.predict_proba(true_dev)
  dev_preds = model.predict_proba(train[num_train:, feature_rows])
  dev_truth = train[num_train:, :full_dim]
  true_dev_proxy = train[num_train:, :full_dim].copy().astype(np.float64)
  #true_dev_proxy[:, proxy_i] = dev_preds[:, 0]
  true_dev_proxy_dist = get_fractional_dist(true_dev_proxy, columns, proxy_var)

  dev_proxy_dist, true_errs = construct_proxy_dist(true_dev_proxy_dist, .2, proxy_var, .005, nondiff = False)

  new_data_array = np.copy(true_dev_proxy)  # Make a copy to store modified data


  for i in range(true_dev_proxy.shape[0]):
    combination = tuple(true_dev_proxy[i])
    error_prob = true_errs.dict.get(combination, 0.0)  # Get the error probability, defaulting to 0 if not found
    if random.random() < error_prob:
      # If the random number is less than the error probability, switch U to the opposite value
      new_data_array[i, proxy_i] = 1 - true_dev_proxy[i, proxy_i]

  true_dev_proxy = new_data_array

  train_accuracy = (accuracy_score(model.predict(train[num_train:, feature_rows]), train[num_train:, proxy_i]))
  print(train_accuracy)
  # import pdb; pdb.set_trace()

  # test_features = np.concatenate((test[:, :proxy_i], test[:, 1 + proxy_i:]),
  #                                axis=1)
  # test_preds = model.predict_proba(test_features)
  test_preds = model.predict_proba(test[:, feature_rows])
  test_proxy = test[:, :full_dim].copy().astype(np.float64)
  # test_proxy[:, proxy_i] = test_preds[:, 0]

  experiment_data.true_err = true_errs

  new_data_array = np.copy(test_proxy)  # Make a copy to store modified data


  for i in range(test_proxy.shape[0]):
    combination = tuple(test_proxy[i])
    error_prob = true_errs.dict.get(combination, 0.0)  # Get the error probability, defaulting to 0 if not found
    if random.random() < error_prob:
      # If the random number is less than the error probability, switch U to the opposite value
      new_data_array[i, proxy_i] = 1 - test_proxy[i, proxy_i]

  test_proxy = new_data_array
  true_test_proxy_dist = get_fractional_dist(test_proxy, columns, proxy_var)

  accuracy = accuracy_score(model.predict(test[:, feature_rows]), test[:, proxy_i])
  print(accuracy)

  print(full_dim)


  correct_distribution = correct_with_model(experiment_data, true_test_proxy_dist, proxy_var, true_dev_proxy[:, :full_dim], dev_truth, influence_vars = influence_vars)
  return [correct_distribution], true_dev_proxy_dist, accuracy





def check_restoration(dist1, dist2):
  '''
  Calculate the L2 distance between two distributions as a sanity check.
  '''
  dist_err = 0
  assert dist1.keys() == dist2.keys()
  for key in dist1:
    dist_err += (dist1[key] - dist2[key]) ** 2

  return dist_err


def correct(proxy_dist, proxy_var, err_ranges,
            nondiff=False, sample_err_rates=0, debug=False):
  new_dists = []
  if sample_err_rates > 0:
    err_keys, err_params = list(zip(*sorted(err_ranges.dict.items())))
    for _ in range(sample_err_rates):
      err_vals = []
      for (p, n) in err_params:
        if n > 0:
          err_vals.append(np.random.binomial(n, p) / n)
        else:
          err_vals.append(0)
      # err_vals = [np.random.binomial(n, p) / n for (p, n) in err_params]
      err_dist = dict(zip(itertools.product(*[range(2) for _ in proxy_dist.columns]), err_vals))
      err_dist = Distribution(data=err_dist, columns=proxy_dist.columns, normalized=False)
      new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
      new_dists.append(new_dist)

  else:
    err_keys, err_boundaries = list(zip(*sorted(err_ranges.dict.items())))
    if nondiff:
      nondiff_boundaries = list(set(err_boundaries))
      for nondiff_err_vals in itertools.product(*nondiff_boundaries):
        boundary_assn = dict(zip(nondiff_boundaries, nondiff_err_vals))
        assn = tuple(itertools.product(*[range(2) for _ in proxy_dist.columns]))
        err_dist = {a: boundary_assn[err_ranges.dict[a]] for a in assn}
        err_dist = Distribution(data=err_dist, columns=proxy_dist.columns, normalized=False)
        new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
        new_dists.append(new_dist)

    else:
      if len(err_boundaries) > 16 and len(err_boundaries[0]) > 1:
        raise ValueError("{} is too many".format(2 ** len(err_boundaries)))
      for err_vals in itertools.product(*err_boundaries):
        err_dist = dict(zip(itertools.product(*[range(2) for _ in proxy_dist.columns]), err_vals))
        err_dist = Distribution(data=err_dist, columns=proxy_dist.columns, normalized=False)
        new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
        new_dists.append(new_dist)
  return new_dists


# @profile
def fractional_impute_and_correct(experiment_data, train, test, columns, proxy_var,
                                  train_percent=0.5, c_dim=1, u_dim=1,
                                  sample_err_rates=0, bootstrap=1,
                                  nondiff=False, alpha=None, debug=False, influence_vars = None):
  '''
  train: training data
  test: testing data
  num_train: how many examples of training data to use for training,
    (leaving the rest for development)
  proxy_i: what is the index of the proxied variable (e.g. 1)
  confound_i: what are the indices of the proxy's confounders (e.g. 0, 2)
  '''


  # features are everything but the proxy variable
  full_dim = len(columns)
  proxy_i = columns.index(proxy_var)

  feature_rows = tuple(i for i in range(train.shape[1]) if i != proxy_i)
  # train_features = np.concatenate((train[:, :proxy_i], train[:, 1 + proxy_i:]),
                                  # axis=1)

  num_train = int(train_percent * train.shape[0])
  num_dev = train.shape[0] - num_train

  model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
  # model.fit(train_features[:num_train, :], train[:num_train, proxy_i])
  model.fit(train[:num_train, feature_rows], train[:num_train, proxy_i])

  # true_dev = train_features[num_train:, :]
  # dev_preds = model.predict_proba(true_dev)
  dev_preds = model.predict_proba(train[num_train:, feature_rows])
  dev_truth = train[num_train:, :full_dim]
  true_dev_proxy = train[num_train:, :full_dim].copy().astype(np.float64)
  #true_dev_proxy[:, proxy_i] = dev_preds[:, 0]
  true_dev_proxy_dist = get_fractional_dist(true_dev_proxy, columns, proxy_var)
  experiment_data.set_p_dot(true_dev_proxy_dist.dict)

  dev_proxy_dist, true_errs = construct_proxy_dist(true_dev_proxy_dist, .2, proxy_var, .005, nondiff = False)

  new_data_array = np.copy(true_dev_proxy)  # Make a copy to store modified data


  for i in range(true_dev_proxy.shape[0]):
    combination = tuple(true_dev_proxy[i])
    error_prob = true_errs.dict.get(combination, 0.0)  # Get the error probability, defaulting to 0 if not found
    if random.random() < error_prob:
      # If the random number is less than the error probability, switch U to the opposite value
      new_data_array[i, proxy_i] = 1 - true_dev_proxy[i, proxy_i]

  true_dev_proxy = new_data_array
  # test_features = np.concatenate((test[:, :proxy_i], test[:, 1 + proxy_i:])
  #                                axis=1)
  # test_preds = model.predict_proba(test_features)
  test_preds = model.predict_proba(test[:, feature_rows])
  test_proxy = test[:, :full_dim].copy().astype(np.float64)
  #test_proxy[:, proxy_i] = test_preds[:, 0]

  experiment_data.true_err = true_errs

  new_data_array = np.copy(test_proxy)  # Make a copy to store modified data


  for i in range(test_proxy.shape[0]):
    combination = tuple(test_proxy[i])
    error_prob = true_errs.dict.get(combination, 0.0)  # Get the error probability, defaulting to 0 if not found
    if random.random() < error_prob:
      # If the random number is less than the error probability, switch U to the opposite value
      new_data_array[i, proxy_i] = 1 - test_proxy[i, proxy_i]

  test_proxy = new_data_array
  true_dist = get_fractional_dist(train[num_train:, :full_dim].copy().astype(np.float64), columns, proxy_var)

  new_dists = []
  for _ in range(bootstrap):
    if bootstrap > 1:
      indices = np.random.choice(np.arange(num_dev), size=num_dev)
      dev_proxy = true_dev_proxy[indices, :]
      dev = dev_truth[indices, :]
    else:
      dev_proxy = true_dev_proxy
      dev = dev_truth

    err_ranges = calculate_error_matrix(experiment_data,
        dev_proxy, dev, proxy_var, columns,
        sample=sample_err_rates > 0,
        alpha=alpha, nondiff=nondiff, debug=debug, influence_vars = influence_vars)

    if debug:
      test_err = calculate_error_matrix(experiment_data,
          test_proxy, test[:, :full_dim], proxy_var, columns,
          sample=sample_err_rates > 0,
          alpha=alpha, nondiff=nondiff, debug=debug, influence_vars = influence_vars)
      print("train_err:", {key: [round(x, 3) for x in val]
                           for key, val in err_ranges.dict.items()})
      print("test_err:", {key: round(val[0], 3)
                          for key, val in test_err.dict.items()})

    proxy_dist = get_fractional_dist(test_proxy, columns, proxy_var)
    dists = correct(proxy_dist, proxy_var, err_ranges,
                    nondiff=nondiff, sample_err_rates=sample_err_rates,
                    debug=debug)
    new_dists.extend(dists)

  return new_dists, true_dev_proxy_dist


def train_adjust(experiment_data, train, test, proxy_var, c_dim=1, u_dim=1,
                 bootstrap=1, sample_err_rates=0, alpha=None,
                 train_percent=0.5, nondiff=False, debug=False, influence_vars = None, impute_func = None, method_type = None):
  '''
  Given train and test data, train a logistic regression classifier to
    impute a proxy for the missing variables, then calculate the errors
    from an oracle in causal effect estimation.
  '''
  full_dim = 2 + c_dim + u_dim
  truth = test[:, :full_dim]
  columns = ['a', 'y', *['c{}'.format(i) for i in range(c_dim)],
             *['u{}'.format(i) for i in range(u_dim)]]
  test_dist = get_dist(truth, columns)
  oracle_effect = gformula(test_dist)

  if method_type is not None:
    new_dists, proxy, accuracy = impute_func(experiment_data,
        train, test, columns, proxy_var,
        c_dim=c_dim, u_dim=u_dim, train_percent=train_percent,
        sample_err_rates=sample_err_rates, bootstrap=bootstrap, nondiff=nondiff,
        alpha=alpha, debug=debug, influence_vars = influence_vars)
    return test_dist, new_dists, proxy, accuracy
  else:
    new_dists, proxy = impute_func(experiment_data,
        train, test, columns, proxy_var,
        c_dim=c_dim, u_dim=u_dim, train_percent=train_percent,
        sample_err_rates=sample_err_rates, bootstrap=bootstrap, nondiff=nondiff,
        alpha=alpha, debug=debug, influence_vars = influence_vars)
    return test_dist, new_dists, proxy

  # print("have {} new dists".format(len(new_dists)))


  # assert type(proxy) == Distribution
  # misspecified_effect = gformula(proxy)

  # corrected_effects = []
  # for new_dist in new_dists:
  #   corrected_effect = gformula(new_dist)
  #   corrected_effects.append(corrected_effect)

  # if debug:
  #   print("oracle: {:.3f}".format(oracle_effect))
  #   print("uncorrected: {:.3f}".format(misspecified_effect))

  #   print("corrected:")
  #   table = OrderedDict()
  #   table['min'] = min(corrected_effects)
  #   table['p25'] = np.percentile(corrected_effects, 25)
  #   table['mean'] = np.mean(corrected_effects)
  #   table['p75'] = np.percentile(corrected_effects, 75)
  #   table['max'] = max(corrected_effects)

  #   width = max(7, max(len(x) for x in table.keys()))
  #   print("".join(["{:^{width}s}".format(key, width=width) for key in table.keys()]))
  #   print("".join(["{:{width}.3f}".format(val, width=width) for val in table.values()]))



# @profile
def synthetic(experiment_data, n_examples, n_train, seed=None, **kwargs):
  '''
  Run a synthetic experiment using n_examples examples in target dataset p(A*, C, Y)
  and n_train examples of external data to estimate p(A*, A)
  '''

  nondiff = kwargs.get('nondiff', False)
  c_dim = kwargs.get('c_dim', 1)
  u_dim = kwargs.get('u_dim', 1)
  ay_effect = kwargs.get('ay_effect', 0.1)
  dist_seed = kwargs.get('dist_seed')
  nondiff_text = ",".join(kwargs.get('influence_vars'))
  print(nondiff_text)
  sampler = SyntheticData(c_dim=c_dim, u_dim=u_dim, nondiff_text=nondiff_text,
                          topic_std=0.075,
                          ay_effect=ay_effect, seed=dist_seed)
  proxy_var = 'u0'

  if seed is not None:
    np.random.seed(seed)

  # sampling 100k examples for each takes about 1min
  truth = sampler.sample_truth(n_examples)
  truth_t = sampler.sample_text(truth)
  external = sampler.sample_truth(n_train)
  external_t = sampler.sample_text(external)

  debug = kwargs.get('debug', False)
  alpha = kwargs.get('alpha', None)
  sample_err_rates = kwargs.get('sample_err_rates', 0)
  bootstrap = kwargs.get('bootstrap', 1)
  train_percent = kwargs.get('train_percent', 0.5)
  influence_vars = kwargs.get('influence_vars')
  impute_func = kwargs.get('impute_func')
  method_type = kwargs.get('method_type')

  train = np.concatenate([external, external_t], axis=1)
  test = np.concatenate([truth, truth_t], axis=1)
  true_dist = sampler.dist

  if method_type is not None:
    test_dist, new_dists, unc_proxy, accuracy = train_adjust(experiment_data,
        train, test, proxy_var,
        c_dim=c_dim, u_dim=u_dim,
        sample_err_rates=sample_err_rates,
        bootstrap=bootstrap,
        train_percent=train_percent,
        nondiff=nondiff,
        alpha=alpha, debug=debug, influence_vars = influence_vars, impute_func = impute_func, method_type = method_type)
  else:
    test_dist, new_dists, unc_proxy = train_adjust(experiment_data,
        train, test, proxy_var,
        c_dim=c_dim, u_dim=u_dim,
        sample_err_rates=sample_err_rates,
        bootstrap=bootstrap,
        train_percent=train_percent,
        nondiff=nondiff,
        alpha=alpha, debug=debug, influence_vars = influence_vars, impute_func = impute_func, method_type = method_type)

  # import pdb; pdb.set_trace()
  if kwargs.get('uncorrected', False):
    return true_dist, [unc_proxy]

  if method_type is not None:
    return true_dist, new_dists, accuracy
  else:
    return true_dist, new_dists


def get_results(test_dist, new_dists, extras=None, unknown_truth=False,
                interval_widths=[90, 95], debug=False):
  if test_dist is None:
    if unknown_truth:
      oracle_effect = 0
    else:
      if debug:
        print("test_dist is None!")
      return {}
  else:
    oracle_effect = gformula(test_dist)

  bad_effects = 0
  corrected_effects = []
  for new_dist in new_dists:
    corrected_effect = gformula(new_dist)
    if -1 < corrected_effect < 1:
      corrected_effects.append(corrected_effect)
    else:
      bad_effects += 1

  if bad_effects > 0:
    print("{} effects with effects outside [-1, 1]")

  if len(corrected_effects) == 0:
    print("no corrected results returned")
    return {}

  corrected_effects = np.array(corrected_effects, dtype=np.float32)
  d = defaultdict(float)

  covered = np.min(corrected_effects) <= oracle_effect <= np.max(corrected_effects)
  d['covered'] = float(covered)

  width = min(1, np.max(corrected_effects)) - max(-1, np.min(corrected_effects))
  d['width'] = width

  percentile_keys = [*[(100 - w) / 2 for w in interval_widths],
                     *[w + (100 - w) / 2 for w in interval_widths]]
  percentile_vals = np.percentile(corrected_effects, percentile_keys,
                                  interpolation='nearest')
  percentiles = dict(zip(percentile_keys, percentile_vals))
                    
  for p, val in percentiles.items():
    d['p{}'.format(p)] = val - oracle_effect

  for w in interval_widths:
    low = percentiles[(100 - w) / 2]
    high = percentiles[w + (100 - w) / 2]
    d['trunc{}width'.format(w)] = min(1, high) - max(-1, low)
    truncated_mean = np.mean(corrected_effects[np.where(np.logical_and(
        low < corrected_effects, corrected_effects < high))])
    d['trunc{}mean'.format(w)] = truncated_mean - oracle_effect

    low_overlap = oracle_effect - low
    high_overlap = high - oracle_effect
    # bigger is better for both of these; take worst
    d['trunc{}overlap'.format(w)] = min(high_overlap, low_overlap)

    d['trunc{}cov'.format(w)] = float(low < oracle_effect < high)

  d['abs_mean'] = np.abs(np.mean(corrected_effects) - oracle_effect)
  d['mean'] = np.mean(corrected_effects) - oracle_effect
  d['median'] = np.median(corrected_effects) - oracle_effect
  d['max'] = min(1, np.max(corrected_effects)) - oracle_effect
  d['min'] = max(-1, np.min(corrected_effects)) - oracle_effect

  d['raw_oracle_effect'] = oracle_effect
  d['raw_mean_effect'] = np.mean(corrected_effects)

  if extras is not None:
    for key in extras:
      d[key] = extras[key]

  return d


def get_outfn(args):
  base = "me_{}".format(args.logn_examples)
  seeds = "-{}-{}-{}".format(args.dist_seed, args.exp_seed, args.k)
  return "{}{}.json".format(base, seeds)

class Experimental_Data:
  def __init__(self):
      self.model_coef = None
      self.model_int = None
      self.err_matrix = None
      self.p_dot = None
      self.true_err = None
      self.model = None

  def set_model_coef(self, coef):
      self.model_coef = coef

  def set_model_int(self, intercept):
    self.model_int = intercept

  def set_err_matrix(self, matrix):
    self.err_matrix = matrix

  def set_p_dot(self, dist):
    self.p_dot = dist

  def get_p_dot(self):
    print(sum(self.p_dot.values()))
    print(dict(zip(self.p_dot.keys(), self.p_dot.values())))

  def get_err_matrix(self):
    print(self.err_matrix.dict)

  def get_model_params(self):
    print(self.model_coef)
    print(self.model_int)


# @profile
def main(method = None, influencers = None, cdim = None):
  # seed, method = None, influencers = None
  parser = argparse.ArgumentParser()
  parser.add_argument("--logn_examples", type=float, default = 4, help="how many examples (log 10)")

  parser.add_argument("--k", type=int, default=4, help="how many runs for each?")
  parser.add_argument("--dataset", type=str, default='synthetic')
  parser.add_argument("--min_freq", type=int, default=10,
                      help="min freq for yelp data vocabulary")
  parser.add_argument("--alpha", type=float, default=None, help="alpha for conf int")
  parser.add_argument("--sample_err_rates", type=int, default=0,
                      help="error rates to sample")
  parser.add_argument("--bootstrap", type=int, default=1,
                      help="bootstrap sample")
  parser.add_argument("--vocab_size", type=int, default=4334,
                      help="vocab size for synthetic data")
  parser.add_argument("--n_vocab", type=int, default=100000,
                      help="how many examples to use to build vocab")
  parser.add_argument("--logn_train", type=float, default=-1,
                      help="how many examples (log 10) to train the classifier")
  parser.add_argument("--train_percent", type=float, default=0.5,
                      help="how much of train set to train (vs estimate err)")
  parser.add_argument("--ay_effect", type=float, default=0.1,
                      help="true effect of a on y")
  parser.add_argument("--c_dim", type=int, default=1,
                      help="dimensionality of c")
  parser.add_argument("--u_dim", type=int, default=1,
                      help="dimensionality of u")
  parser.add_argument("--dist_seed", type=int, default=4)
  parser.add_argument("--exp_seed", type=int, default=4)
  parser.add_argument("--write", type=str, default='append')
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--nondiff", action='store_true')
  parser.add_argument("--uncorrected", action='store_true')
  # parser.add_argument("--workdir", type=str, default='work/')
  parser.add_argument("--outdir", type=str, default="json/")

  parser.add_argument("--influence_vars", type = str, default = None)

  parser.add_argument("--method_type", type = str, default = None)

  args = parser.parse_args()

  # args.dist_seed = seed
  # args.exp_seed = seed

  if method is not None:
    args.method_type = "new"

  if influencers is not None:
    args.influence_vars = influencers

  if args.influence_vars is not None:
    args.influence_vars = args.influence_vars.split(",")

  if cdim is not None:
    args.c_dim = cdim

  if args.method_type is not None:
    impute_func = impute_and_correct_with_model
  else:
    impute_func = fractional_impute_and_correct

  n_examples = int(10 ** args.logn_examples)
  if args.logn_train < 0:
    # n_train = max(1000, n_examples // 10)
    n_train = n_examples
  else:
    n_train = int(10 ** args.logn_train)
  if args.debug:
    print("measure", n_examples, n_train, args.k, args.dataset, args.min_freq)

  outfn = "me.{}.{}.{}.{}.json".format(args.dataset, args.logn_examples, args.k, args.min_freq)
  if not os.path.exists(args.outdir):
    logging.error("{} doesn't exist, quitting".format(args.outdir))
    return
  if os.path.exists(os.path.join(args.outdir, outfn)):
    logging.error("{} already exists, quitting".format(outfn))
    return

  job_args = {'debug': args.debug, 'alpha': args.alpha, 'u_dim': args.u_dim,
              'uncorrected': args.uncorrected,
              'train_percent': args.train_percent,
              'dist_seed': args.dist_seed, 'c_dim': args.c_dim,
              'bootstrap': args.bootstrap, 'ay_effect': args.ay_effect,
              'sample_err_rates': args.sample_err_rates, 'nondiff': args.nondiff, 'influence_vars' : args.influence_vars, 'impute_func' : impute_func,
              'method_type' : args.method_type}
  
  
  if args.dataset == 'synthetic':
    test_func = synthetic
    job_args['vocab_size'] = args.vocab_size
  else:
    raise ValueError("unknown dataset {}".format(args.dataset))

  np.random.seed(args.exp_seed)
  iinfo = np.iinfo(np.int32)
  exp_seeds = np.random.randint(0, iinfo.max, size=args.k)

  errors = defaultdict(list)
  raw_results = []
  classifier_rates = []
  experiment_data = Experimental_Data()
  for i, seed in enumerate(exp_seeds):
    if args.debug:
      print(" {} ".format(i), end='\r')
    try:
      if args.method_type is not None:
        truth, estimates, accuracy = test_func(experiment_data, n_examples, n_train, seed, **job_args)
        classifier_rates.append(1 - accuracy)
        raw_results.append(get_results(truth, estimates, interval_widths=[90, 95]))
      else:
        truth, estimates = test_func(experiment_data, n_examples, n_train, seed, **job_args)
        raw_results.append(get_results(truth, estimates, interval_widths=[90, 95]))
    except Exception as e:
      if args.debug:
        raise e
      errors[type(e)].append(str(e))
      pass

  # experiment_data.get_p_dot()



  if len(errors) > 0:
    print(errors)
  keysets = [set(res.keys()) for res in raw_results]
  all_keys = set("-".join(sorted(str(key) for key in keyset))
                 for keyset in keysets)
  max_keyset = max(keysets)
  percent_max = np.mean([1 if k == max_keyset else 0 for k in keysets])
  print("We have {} results with {} ({:.1f}% max) keysets from {} tries".format(
      len(raw_results), len(all_keys), 100 * percent_max, args.k))
  keys = raw_results[0].keys()
  results = defaultdict(list)
  for result in raw_results:
    if set(result.keys()) == max_keyset:
      for key in max_keyset:
        results[key].append(result[key])
  results = {key: np.array(val) for key, val in results.items()}

  means = {key: np.mean(result) for key, result in results.items()}
  mean_abs = {key: np.mean(np.abs(result))
                for key, result in results.items()}
  stds = {key: np.std(result) for key, result in results.items()}
  print("min/p5/mean/p95/max: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(
      means.get('min', np.nan),
      means.get('p5.0', np.nan), means.get('mean', np.nan),
      means.get('p95.0', np.nan), means.get('max', np.nan)))

  print("standard deviation: {:.5f}".format(
    stds.get('min', np.nan)))
  

  outfn = os.path.join(args.outdir, get_outfn(args))

  if args.write == 'overwrite':
    mode = 'w'
  else:
    mode = 'a'
  with open(outfn, mode) as outf:
    obj = vars(args)
    for key in keys:
      obj['mean_{}'.format(key)] = means[key]
      obj['std_{}'.format(key)] = stds[key]
      obj['mean_abs_{}'.format(key)] = mean_abs[key]
    outf.write("{}\n".format(json.dumps(obj, cls=NumpySerializer)))

  return [means.get('min', np.nan_to_num), stds.get('min', np.nan), np.mean(np.array(classifier_rates)), experiment_data]

if __name__ == "__main__":
  influencer = 'a,y,c0,c1,c2'

  influence_size = []
  error_matrix_data = []
  error_matrix_std = []
  model_data = []
  model_std = []
  classifier_rates = []

  for i in range(3, 4):
    matrix_list = main(influencers = influencer, cdim = i)
    model_list = main(method = "new", influencers = influencer, cdim = i)

    model_experiment = model_list[3]
    matrix_experiment = matrix_list[3]

    # print(matrix_experiment.true_err.dict.values())
    # print(matrix_experiment.err_matrix.dict.values())

    matrix_true_errs = [matrix_experiment.true_err.dict[key] for key in matrix_experiment.err_matrix.dict]
    matrix_pred_errs = [matrix_experiment.err_matrix.dict[key] for key in matrix_experiment.err_matrix.dict]

    matrix_true_errs = np.array(matrix_true_errs)
    matrix_pred_errs = np.array(matrix_pred_errs).flatten()

    model_true_errs = [model_experiment.true_err.dict[key] for key in matrix_experiment.err_matrix.dict]
    model_pred_errs = []

    for assn in model_experiment.p_dot.keys():
      print(model_experiment.model.predict_proba(np.array(assn).reshape(1, -1)))
      model_pred_errs.append(model_experiment.model.predict_proba(np.array(assn).reshape(1, -1))[0][1])

    print(model_pred_errs)
    print(model_true_errs)

    


    # absolute_differences = [abs(((1.0,) * len(matrix_experiment.err_matrix.dict) - (matrix_experiment.err_matrix.dict[key])) 
    #                             - (matrix_experiment.true_err.dict[key])) for key in matrix_experiment.err_matrix.dict]


    #to_plot = dict(zip(matrix_experiment.p_dot.values(), np.array(list(matrix_experiment.err_matrix.dict.values())) - np.array(list(matrix_experiment.true_err.dict.values()))))

    fig2 = plt.figure()
    plt.scatter(list(matrix_experiment.p_dot.values()), abs((np.ones(len(matrix_true_errs)) - matrix_pred_errs) - matrix_true_errs))
    plt.scatter(list(model_experiment.p_dot.values()), abs(np.array(model_pred_errs) - np.array(model_true_errs)))
    plt.title("Matrix error rates versus probability distribution of assignment")
    plt.xlabel("Probability of assignment")
    plt.ylabel("Error rate of assignment")
    plt.show()

  


  for i in range(1, 2):
    matrix_list = main(influencers = influencer, cdim = i)
    model_list = main(method = "new", influencers = influencer, cdim = i)
    error_matrix_data.append(abs(matrix_list[0]))
    error_matrix_std.append(matrix_list[1])
    model_data.append(abs(model_list[0]))
    model_std.append(model_list[1])
    classifier_rates.append(model_list[2])
    influencer += ",c" + str(i)
    influence_size.append(2 + i)
    model_experiment = model_list[3]
    matrix_experiment = matrix_list[3]



  fig1 = plt.figure()
  plt.plot(influence_size, error_matrix_data, color = 'red', label = 'Error matrix data')
  plt.plot(influence_size, model_data, color = 'blue', label = 'Model data')

  plt.plot(influence_size, np.clip([a - b for a,b in zip(error_matrix_data, error_matrix_std)], 0, None), linestyle = '--', color = 'red')
  plt.plot(influence_size, np.clip([a - b for a,b in zip(model_data, model_std)], 0, None), linestyle = '--', color = 'blue')

  plt.plot(influence_size, [a + b for a,b in zip(error_matrix_data, error_matrix_std)], linestyle = '--', color = 'red')
  plt.plot(influence_size, [a + b for a,b in zip(model_data, model_std)], linestyle = '--', color = 'blue')

  plt.legend()
  plt.title("Comparison of measurement error correction performance")
  plt.xlabel("# of influencing variables")
  plt.ylabel("Corrected causal estimation error rate")

  fig2 = plt.figure()
  plt.plot(influence_size, classifier_rates)
  plt.title("Error rates of U* classifier on test data")
  plt.xlabel("# of influencing variables")
  plt.ylabel("Classifier error rate")

  plt.show()

