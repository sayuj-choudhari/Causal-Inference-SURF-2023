# Code from causal_sensitivity_2020 repository by @zachwooddoughty

# import pdb; pdb.set_trace()

import json
import itertools
import math
import numpy as np
import sklearn.datasets
import sklearn.linear_model

from collections import defaultdict


def get_dist(arr, columns, debug=False):
  '''
  Calculate the probability mass on all variable assignments
  '''
  assert type(arr) == np.ndarray

  full_dim = len(columns)
  n = arr.shape[0]
  d = {}
  for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
    where = [arr[:, i] == assn[i] for i in range(len(assn))]
    where = np.all(np.stack(where, axis=1), axis=1)
    d[tuple(assn)] = np.sum(where) / n

  return Distribution(data=d, columns=columns)


def get_fractional_dist(proxy_arr, columns, proxy_var, debug=False):
  '''
  '''

  assert type(proxy_arr) == np.ndarray

  full_dim = len(columns)
  n = proxy_arr.shape[0]
  d = defaultdict(float)

  proxy_i = columns.index(proxy_var)
  non_proxies = list(set(range(full_dim)) - set([proxy_i]))
  for assn in itertools.product(*[range(2) for _ in range(len(non_proxies))]):
    where = [proxy_arr[:, non_proxies[i]] == assn[i] for i in range(len(assn))]
    where = np.all(np.stack(where, axis=1), axis=1)
    proba = proxy_arr[where, :][:, proxy_i]

    assn0 = list(assn)
    assn0.insert(proxy_i, 0)
    assn1 = list(assn)
    assn1.insert(proxy_i, 1)

    d[tuple(assn0)] = np.sum(proba) / n
    d[tuple(assn1)] = np.sum(1 - proba) / n

  return Distribution(data=dict(d), columns=columns)

def construct_model_proxy_dist(test_dist, classifier_error,
                               proxy_var, spec_var, nondiff=True):

    assert nondiff

    full_dim = len(test_dist.columns)
    proxy_i = test_dist.columns.index(proxy_var)

    true_errs = {}

    coef = np.random.uniform(-2, 2, full_dim)

    for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
        true_errs[assn] = np.array(assn).dot(coef)

    errs = list(true_errs.values())
    key_list = list(true_errs.keys())

    adjusted_errs = [np.sqrt(spec_var / np.var(errs)) * element for element in errs]
    adjusted_errs = adjusted_errs - np.mean(adjusted_errs) + classifier_error
    adjusted_errs = np.clip(adjusted_errs, 0.01, 0.99)

    # print(f"proxy dist has err mean {np.mean(adjusted_errs):.3f} and var {np.var(adjusted_errs):.3f}")

    true_errs = dict(zip(key_list, adjusted_errs))

    return None, true_errs

def construct_proxy_dist(test_dist, classifier_error,
                         proxy_var, spec_var, nondiff=False):
  # Construct the true error rates of the proxy distribution
  # not_proxy_cols = [col for col in test_dist.columns if col != proxy_var]
  # not_proxy_marg = test_dist.get_marginal(not_proxy_cols)
  # err_dim = len(not_proxy_cols)

  full_dim = len(test_dist.columns)
  proxy_i = test_dist.columns.index(proxy_var)

  true_errs = {}

  if nondiff:
    proxy_marg = test_dist.get_marginal([proxy_var])

    err_margin = np.random.normal(0, classifier_error / 2)
    err0 = classifier_error - err_margin / 2
    err1 = classifier_error + err_margin / 2

    total_err = err0 * proxy_marg.get(**{proxy_var: 0}) + err1 * proxy_marg.get(**{proxy_var: 1})
    err0 = err0 * classifier_error / total_err
    err1 = err1 * classifier_error / total_err

    # print("true errs", err0, err1)
    
    for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
      if assn[proxy_i] == 0:
        err = err0
      else:
        err = err1
      true_errs[assn] = err
  else:
    # combos = list(itertools.product(*[range(2) for _ in range(full_dim)]))
    # influence_vars = []
    # indices = [i for i in range(len(test_dist.columns)) if test_dist.columns[i] in influence_vars]
    # for assn in itertools.product(*[range(2) for _ in range(len(indices))]):
    #   err = np.random.normal(classifier_error, classifier_error / 4)
    #   index_vals = dict(zip(indices, assn))
    #   for combo in combos:
    #     y = combo
    #     x = [n for n in range(len(combo))]
    #     if all(item in dict(zip(x, y)).items() for item in index_vals.items()):
    #       true_errs[combo] = err

    combos = list(itertools.product(*[range(2) for _ in range(full_dim)]))
    diff_level = full_dim
    indices = [i for i in range(diff_level)]
    for assn in itertools.product(*[range(2) for _ in range(len(indices))]):
      err = np.random.normal(classifier_error, spec_var)
      index_vals = dict(zip(indices, assn))
      removal_list = []
      for combo in combos:
        y = combo
        x = [n for n in range(len(combo))]

        if all(item in dict(zip(x, y)).items() for item in index_vals.items()):
          true_errs[combo] = err
          removal_list.append(combo)

      for m in range(len(removal_list)):
         combos.remove(removal_list[m])


    errs = list(true_errs.values())
    key_list = list(true_errs.keys())



  adjusted_errs = [np.sqrt(spec_var / np.var(errs)) * element for element in errs]
  adjusted_errs = adjusted_errs - np.mean(adjusted_errs) + classifier_error
  adjusted_errs = np.clip(adjusted_errs, 0.01, 0.99)

  print(f"proxy dist has err mean {np.mean(adjusted_errs):.3f} and var {np.var(adjusted_errs):.3f}")

  true_errs = dict(zip(key_list, adjusted_errs))

  proxy_dist = {}
  for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
    # Choose the right error rates
    err0 = true_errs[assn]
    proxy_assn = dict(zip(test_dist.columns, assn))
    proxy_dist[assn] = (1 - err0) * test_dist.get(**proxy_assn)

    error_assn = proxy_assn.copy()
    error_assn[proxy_var] = 1 - proxy_assn[proxy_var]
    err1 = true_errs[tuple(error_assn[col] for col in test_dist.columns)]
    proxy_dist[assn] += err1 * test_dist.get(**error_assn)

  proxy_dist = Distribution(data=proxy_dist, columns=test_dist.columns)
  true_errs = Distribution(data=true_errs, columns=test_dist.columns,
                           normalized=False)
  return proxy_dist, true_errs


def construct_model_proxy_dist(test_dist, classifier_error,
                               proxy_var, spec_var, nondiff=True):

    assert nondiff

    full_dim = len(test_dist.columns)
    proxy_i = test_dist.columns.index(proxy_var)

    true_errs = {}

    coef = np.random.uniform(-2, 2, full_dim)

    for assn in itertools.product(*[range(2) for _ in range(full_dim)]):
        true_errs[assn] = np.array(assn).dot(coef)

    errs = list(true_errs.values())
    key_list = list(true_errs.keys())

    adjusted_errs = [np.sqrt(spec_var / np.var(errs)) * element for element in errs]
    adjusted_errs = adjusted_errs - np.mean(adjusted_errs) + classifier_error
    adjusted_errs = np.clip(adjusted_errs, 0.01, 0.99)

    # print(f"proxy dist has err mean {np.mean(adjusted_errs):.3f} and var {np.var(adjusted_errs):.3f}")

    true_errs = dict(zip(key_list, adjusted_errs))

    # TODO can add in computation for proxy_dist
    return None, true_errs



class Distribution:
  def __init__(self, data=None, columns=None, proxy_var=None, normalized=True):

    assert data is not None and columns is not None, \
        "data and columns keywords required"
    self.columns = columns
    self.full_dim = len(columns)
    self.normalized = normalized
    self.dict = {}

    assert type(data) == dict
    for assn in itertools.product(*[range(2) for _ in range(self.full_dim)]):
      assert tuple(assn) in data, assn
    self.dict = data.copy()

    if self.normalized:
      total_weight = math.fsum(list(self.dict.values()))
      assert np.isclose(total_weight, 1, atol=1e-1), total_weight

  def get(self, **kwargs):
    tup = tuple([kwargs[col] for col in self.columns])
    return self.dict[tup]

  def get_marginal(self, keep_cols):
    if not self.normalized:
      raise NotImplementedError("Can't marginalize non-normalized distributions")

    for column in keep_cols:
      assert column in self.columns

    keep_cols = sorted(keep_cols)
    marg_cols = sorted(set(self.columns) - set(keep_cols))
    assert len(keep_cols) > 0
    assert len(marg_cols) > 0

    marginal = defaultdict(float)
    for keep_assn in itertools.product(*[range(2) for col in keep_cols]):
      keep = dict(zip(keep_cols, keep_assn))
      for marg_assn in itertools.product(*[range(2) for col in marg_cols]):
        marg = dict(zip(marg_cols, marg_assn))
        marginal[tuple(keep_assn)] += self.get(**{**keep, **marg})

    marginal_sum = math.fsum(list(marginal.values()))
    assert np.isclose(marginal_sum, 1, atol=1e-1), marginal_sum

    return Distribution(dict(marginal), keep_cols)

  def __truediv__(self, other):
    if not self.normalized:
      raise NotImplementedError("Can't marginalize non-normalized distributions")

    assert type(other) == Distribution
    assert set(self.columns) > set(other.columns)
    new_cols = set(self.columns) - set(other.columns)
    marg_cols = sorted(other.columns)
    assert len(marg_cols) > 0
    assert len(new_cols) > 0
    # print("trying to take p({} | {}) = p({}) / p({})".format(
    #     ",".join(new_cols), ",".join(marg_cols), ",".join(self.columns), ",".join(marg_cols)))

    conditional = defaultdict(float)
    for assn in itertools.product(*[range(2) for col in self.columns]):
      mapping = dict(zip(self.columns, assn))
      numerator = self.get(**mapping)
      marg = {col: mapping[col] for col in marg_cols}
      denominator = other.get(**marg)
      if denominator == 0:
        conditional[tuple(assn)] = 0.
      else:
        conditional[tuple(assn)] = numerator / denominator

    return Distribution(dict(conditional), self.columns, normalized=False)


def dist_pc(dist, c_dim=1, u_dim=1):
  ''' Given a distribution dictionary from get_dist, calculate p(C=1) '''
  pc = 0
  for a in [0, 1]:
    for y in [0, 1]:
      pc += dist[(1, a, y)]

  return pc


class NumpySerializer(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NumpySerializer, self).default(obj)


def maybe_stack(inp):
  ''' If inp is a list or tuple, concatenate along axis 1'''
  if type(inp) in [list, tuple]:
    if any(len(x.shape) == 1 for x in inp):
      inp = [x.reshape(-1, 1) if len(x.shape) == 1 else x for x in inp]
      return np.concatenate(inp, axis=1)

    return np.concatenate(inp, axis=1)

  if len(inp.shape) == 1:
    return inp.reshape(-1, 1)
  return inp


def naive(truth):
  '''
  Given a (c, a, y) dataset of confounder, treatment, outcome
  Calculate the causal effect assuming no confounding or mismeasurement
    as p(Y=1 | A=1) - p(Y=1 | A=0).
  '''
  v = np.stack(truth, axis=0)

  tot_effect = 0
  for a in [0, 1]:
    where = (v[1, :] == a)
    y_true = np.sum(v[:, np.where(where)][2, :])
    y_tot = np.sum(where)
    tot_effect += (-1, 1)[a] * (y_true / y_tot)

  return tot_effect


def fit_bernoulli(v):
  ''' Calculate p(C=1) for a binary variable '''
  assert len(v.shape) == 1 or v.shape[1] == 1
  return np.sum(v) / v.shape[0]


def fit_simple(inp, out):
  '''
    inp is a n x k matrix of features (e.g. c, a)
    out is a n x 1 matrix of targets (e.g. y)
    Simply calculate the 2 ** k truth table probabilities
  '''
  inp = maybe_stack(inp)
  n = inp.shape[0]
  assert out.shape[0] == n
  k = inp.shape[1]

  dist = {}
  assns = list(itertools.product(*[range(2) for _ in range(k)]))
  for assn in assns:
    where = [inp[:, i] == assn[i] for i in range(len(assn))]
    where = np.all(np.stack(where, axis=0), axis=0)
    y_true = np.sum(out[where])
    y_tot = np.sum(where)
    dist[tuple(assn)] = y_true / max(1, y_tot)

  return dist


def fit_logis(inp, out):
  '''
    inp is a n x k matrix of features (e.g. c, y, t_i)
    out is a n x 1 matrix of targets (e.g. a)
    Fit a logistic regression classifier to predict the target
  '''
  inp = maybe_stack(inp)
  n = inp.shape[0]
  assert out.shape[0] == n

  model = sklearn.linear_model.LogisticRegression(C=1e8)
  model.fit(inp, out)
  return model


def logis_proba(model, inp):
  '''
    inp is a n x k matrix of features
    model is a logistic regression classifier
    Uses the model to infer the targets
  '''
  inp = maybe_stack(inp)
  return model.predict_proba(inp).reshape(-1)[1]


def gformula(dist, treatment='a', outcome='y'):
  '''
  Use the g-formula to calculate the causal effect of A on Y given C
    Assumes no mismeasurement.
  '''

  assert isinstance(dist, Distribution)
  marg_cols = [col for col in dist.columns if col not in {treatment, outcome}]
  marg = dist.get_marginal(marg_cols)
  not_y_cols = [col for col in dist.columns if col != outcome]
  not_y = dist.get_marginal(not_y_cols)

  p_y_given = dist / not_y

  effect = 0
  for assn in itertools.product(*[range(2) for _ in not_y.columns]):
    mapping = dict(zip(not_y.columns, assn))
    mapping[outcome] = 1
    marg_mapping = {col: mapping[col] for col in marg_cols}
    direction = 2 * mapping[treatment] - 1
    effect += direction * p_y_given.get(**mapping) * marg.get(**marg_mapping)

  return effect


def get_effect_modification(dist, treatment='a', outcome='y',
                       confounder='u0', debug=False):
  assert isinstance(dist, Distribution)

  # for effect modification of u, we adjust with p(c|u) instead of p(c, u)
  marg_cols = [col for col in dist.columns if col not in {treatment, outcome}]
  marg = dist.get_marginal(marg_cols)
  marg = marg / marg.get_marginal([confounder])

  not_y_cols = [col for col in dist.columns if col != outcome]
  not_y = dist.get_marginal(not_y_cols)
  p_y_given = dist / not_y

  effect = 0
  for assn in itertools.product(*[range(2) for _ in not_y.columns]):
    mapping = dict(zip(not_y.columns, assn))
    mapping[outcome] = 1
    marg_mapping = {col: mapping[col] for col in marg_cols}

    # Direction is product of confounder and treatment
    # Want [E[Y(1)|U=1] - E[Y(0)|U=1]] - [E[Y(1)|U=0] - E[Y(0)|U=0]]
    direction = (2 * mapping[treatment] - 1) * (2 * mapping[confounder] - 1)
    effect += direction * p_y_given.get(**mapping) * marg.get(**marg_mapping)

  return effect

  # # simpsons paradox box
  # empty = {0: 0, 1:0}
  # prob_table = defaultdict(lambda: defaultdict(lambda: empty.copy()))
  # counts_table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

  # gender_marginals = defaultdict(lambda: empty.copy())
  # popular_marginals = defaultdict(lambda: empty.copy())

  # total = 2
  # for men in (0, 1):
  #   for popular in (0, 1):
  #     percent_liked = dist[(men, popular, 1)]
  #     percent_not_liked = dist[(men, popular, 0)]
  #     cond_prob = percent_liked / (percent_liked + percent_not_liked)

  #     prob_table[men][popular] = cond_prob
  #     counts_table[men][popular] = (percent_liked + percent_not_liked)

  #     gender_marginals[men][0] += percent_not_liked
  #     gender_marginals[men][1] += percent_liked

  #     popular_marginals[popular][0] += percent_not_liked
  #     popular_marginals[popular][1] += percent_liked
  #     # if debug:
  #     #   print("{} {}: {:.2f} (of {:.2f})".format(
  #     #       "men" if men else "women",
  #     #       "popular" if popular else "not",
  #     #       100 * cond_prob,
  #     #       100 * (percent_liked + percent_not_liked)))

  #     if men == popular:
  #       output += cond_prob
  #     else:
  #       output -= cond_prob

  # for men in (0, 1):
  #   prob_yes = gender_marginals[men][1]
  #   prob_no = gender_marginals[men][0]
  #   cond_prob = prob_yes / (prob_yes + prob_no)

  #   prob_table[men][total] = cond_prob
  #   counts_table[men][total] = (prob_yes + prob_no)

  # total_yes = 0
  # total_no = 0
  # for popular in (0, 1):
  #   prob_yes = popular_marginals[popular][1]
  #   prob_no = popular_marginals[popular][0]

  #   total_yes += prob_yes
  #   total_no += prob_no

  #   cond_prob = prob_yes / (prob_yes + prob_no)
  #   prob_table[total][popular] = cond_prob
  #   counts_table[total][popular] = (prob_yes + prob_no)

  # prob_table[total][total] = total_yes / (total_yes + total_no)
  # counts_table[total][total] = total_yes + total_no

  # if debug:
  #   print_table(prob_table)
  #   print("counts")
  #   print_table(counts_table)


def mcar(truth, prob):
  '''
  Given some truth vector, introduce MCAR missingness with prob.
  '''
  n = truth.shape[0]
  return np.random.binomial(1, prob, n)


def impute(features, targets, test_features, debug=False):
  '''
  Given some training features, training targets, and test features
    Train a logistic model to predict targets from features,
    then use that model to impute test targets from test features.
  '''
  model = fit_logis(features, targets)
  if debug:
    print("acc:", model.score(features, targets))
  test_features = maybe_stack(test_features)
  return model.predict_proba(test_features)


def test_distributions():
  data = {(0, 0): 0.6, (0, 1): 0.2, (1,0): 0.04, (1, 1): 0.16}
  d = Distribution(data=data, columns=['a', 'b'])
  assert d.get(a=0, b=0) == data[(0, 0)]

  pa = d.get_marginal(['a'])
  pba = d / pa
  assert np.isclose(pba.get(a=0, b=0) / pba.get(a=0, b=1),
                    data[(0, 0)] / data[(0, 1)])


if __name__ == "__main__":
  test_distributions()
