# Code from causal_sensitivity_2020 repository by @zachwooddoughty

import argparse
import itertools
import json
import os

from collections import OrderedDict, defaultdict

from utils import gformula, Distribution, NumpySerializer
from utils import get_dist, get_fractional_dist, construct_proxy_dist
from datasets import SyntheticData
from measurement_error import get_results
from measurement_error import get_corrected_dist, calculate_error_matrix

import numpy as np
import sklearn.linear_model
import scipy.optimize


# def solve():




def double_oracle(sampler, classifier_error, corrector_error_width,
                  nondiff=False, debug=False):
  raise NotImplementedError("hasn't been updated for UAI")
  proxy_var = 0
  test_dist = sampler.true_dist()

  corrector_error_offset_mean = corrector_error_width / 3

  proxy_dist, true_errs = construct_proxy_dist(test_dist, classifier_error, proxy_var, nondiff)

  if debug:
    print("truth ", {key: float("{:.4f}".format(val)) for key, val in test_dist.items()})
    print("proxy ", {key: float("{:.4f}".format(val)) for key, val in proxy_dist.items()})

  err_ranges = {}
  for assn in itertools.product(*[range(2) for _ in range(3)]):
    error_offset = np.random.normal(corrector_error_offset_mean, corrector_error_offset_mean)
    error_range_center = classifier_error + error_offset * np.random.choice([-1, 1])
    err_ranges[assn] = [error_range_center - (corrector_error_width / 2),
                        error_range_center + (corrector_error_width / 2)]

  new_dists = []
  err_boundaries = [val for key, val in sorted(err_ranges.items())]
  err_keys = [key for key, _ in sorted(err_ranges.items())]
  for err_vals in itertools.product(*err_boundaries):
    # TODO this needs to be updated?
    errs = {err_keys[i]: err_vals[i] for i in range(len(err_vals))}
    new_dist = get_corrected_dist(proxy_dist, errs, None, proxy_var, confound_i, debug)
    new_dists.append(new_dist)

  if debug:
    printout(test_dist, new_dists)

  return (test_dist, new_dists)


def double_oracle_no_interval(sampler, classifier_error, corrector_error_width,
                              nondiff=False, debug=False):
  raise NotImplementedError("hasn't been updated for UAI")
  proxy_var = 0
  confound_i = (1, 2)
  test_dist = sampler.true_dist()

  proxy_dist, true_errs = construct_proxy_dist(test_dist, classifier_error, proxy_var, nondiff)

  if debug:
    print("truth ", {key: float("{:.4f}".format(val)) for key, val in test_dist.items()})
    print("proxy ", {key: float("{:.4f}".format(val)) for key, val in proxy_dist.items()})

  # we randomly sample a predicted error from a window around the true error
  err_ranges = {}
  for assn in itertools.product(*[range(2) for _ in range(3)]):
    err_ranges[assn] = (np.random.uniform(
        true_errs[assn] - (corrector_error_width / 2),
        true_errs[assn] + (corrector_error_width / 2)), )

  # we now use our proxy dist and our error 'estimate' to calculate the causal effect
  new_dists = []
  err_boundaries = [val for key, val in sorted(err_ranges.items())]
  err_keys = [key for key, _ in sorted(err_ranges.items())]
  for err_vals in itertools.product(*err_boundaries):
    errs = {err_keys[i]: err_vals[i] for i in range(len(err_vals))}
    new_dist = get_corrected_dist(proxy_dist, errs, None, proxy_var, confound_i, debug)
    new_dists.append(new_dist)

  if debug:
    printout(test_dist, new_dists)

  return (test_dist, new_dists)


def uncorrected_classifier(sampler, classifier_error, nondiff=False, debug=False):
  proxy_var = 'u0'
  dist = sampler.dist
  proxy_dist, true_errs = construct_proxy_dist(dist, classifier_error, proxy_var, nondiff)
  return (dist, [proxy_dist])


def classifier_oracle(sampler, classifier_error, n_dev, seed=None,
                      sample_err_rates=0, alpha=None, bootstrap=1,
                      n_func=None,
                      proxy_logits=False, nondiff=False, debug=False):

  assert alpha is None or sample_err_rates == 0

  if seed is not None:
    np.random.seed(seed)

  true_dist = sampler.dist
  columns = true_dist.columns
  proxy_var = 'u0'
  proxy_i = columns.index(proxy_var)
  true_proxy_dist, true_errs = construct_proxy_dist(
      true_dist, classifier_error, proxy_var, spec_var = .1, spec_mean = .2, nondiff=nondiff)

  # # print_table outputs
  # print(gformula(get_corrected_dist(true_proxy_dist, true_errs, proxy_var)))
  # print(true_dist.dict)
  # print(true_errs.dict)
  # print(true_proxy_dist.dict)

  true_dev = np.array(sampler.sample_truth(n_dev))
  true_dev_proxy = true_dev.copy()
  if proxy_logits:
    true_dev_proxy = true_dev_proxy.astype(np.float64)
  for row_i in range(true_dev.shape[0]):
    assn = dict(zip(columns, true_dev[row_i, :].tolist()))
    err_rate = true_errs.get(**assn)
    if not proxy_logits:
      if np.random.binomial(1, err_rate):
        true_dev_proxy[row_i, proxy_i] = 1 - true_dev[row_i, proxy_i]
    else:
      if False:
        val = np.abs(-0.05 + true_dev[row_i, proxy_i])
        if np.random.binomial(1, err_rate):
          true_dev_proxy[row_i, proxy_i] = 1 - val
        else:
          true_dev_proxy[row_i, proxy_i] = val

      else:
        if true_dev[row_i, proxy_i] == 1:
          beta_a, beta_b = 1, 1.5
        else:
          beta_a, beta_b = 1.5, 1
          
        if np.random.binomial(1, err_rate):
          true_dev_proxy[row_i, proxy_i] = np.random.beta(beta_b, beta_a, 1)[0]
        else:
          true_dev_proxy[row_i, proxy_i] = np.random.beta(beta_a, beta_b, 1)[0]

  new_dists = []

  for _ in range(bootstrap):
    if bootstrap > 1:
      indices = np.random.choice(np.arange(n_dev), size=n_dev)
      dev_proxy = true_dev_proxy[indices, :]
      dev = true_dev[indices, :]
    else:
      dev_proxy = true_dev_proxy
      dev = true_dev

    if dev_proxy.dtype == np.int64:
      proxy_dist = get_dist(dev_proxy, columns)
    else:
      proxy_dist = get_fractional_dist(dev_proxy, columns, proxy_var)

    err_ranges = calculate_error_matrix(
        dev_proxy, dev, proxy_var, columns,
        sample=sample_err_rates > 0, n_func=n_func,
        alpha=alpha, nondiff=nondiff, debug=debug)
    # # print_table outputs
    # print(err_ranges.dict)
    # print(proxy_dist.dict)

    if alpha is not None:
      err_contained = []
      for assn, err in true_errs.dict.items():
        assn = dict(zip(columns, assn))
        low, high = err_ranges.get(**assn)
        # print("{:.3f}, {:.3f}, {:.3f}".format(low, err, high))
        err_contained.append(int(low <= err <= high))
      err_contained = np.mean(err_contained)
      print("error intervals cover {:.1f}%".format(100 * err_contained))
    else:
      err_contained = -1

    # calculate the avg corrector error so we can plot it if we want
    if alpha is None:
      no_int_errs = err_ranges
    else:
      no_int_errs = calculate_error_matrix(
          dev_proxy, dev, proxy_var, columns,
          alpha=None, nondiff=nondiff, debug=debug)
    corrector_error = 0
    for key, val in true_errs.dict.items():
      assn = dict(zip(columns, key))
      corrector_error += np.abs(val - no_int_errs.get(**assn)[0])
    corrector_error = np.mean(corrector_error)

    if debug:
      # print("match-up: {:.3f}".format(
      #     np.mean(dev_proxy[proxy_i, :] == dev[proxy_i, :])))
      # print("truth  ", {key: float("{:.4f}".format(val))
      #                   for key, val in true_dist.dict.items()})
        
      print("est proxy ", {key: float("{:.4f}".format(val))
                        for key, val in proxy_dist.dict.items()})
      print("true proxy ", {key: float("{:.4f}".format(val))
                            for key, val in true_proxy_dist.dict.items()})

      print("true errs  ", {key: float("{:.4f}".format(val))
                            for key, val in true_errs.dict.items()})
      print("est errs  ", {key: float("{:.4f}".format(val[0]))
                           for key, val in err_ranges.dict.items()})
      print("corrector error: {:.4f}".format(corrector_error))

    # TODO handle this better
    # if np.any(np.isnan(np.array([x[0] for x in err_ranges.dict.values()]))):
    #   return (None, [], None)

    if np.any(np.isnan(np.array([x[0] for x in err_ranges.dict.values()]))):
      print("NaN in err range!")

    if sample_err_rates > 0:
      err_keys, err_params = list(zip(*sorted(err_ranges.dict.items())))
      for _ in range(sample_err_rates):
        err_vals = [np.random.binomial(n, p) / n for (p, n) in err_params]
        err_dist = dict(zip(itertools.product(*[range(2) for _ in columns]), err_vals))
        err_dist = Distribution(data=err_dist, columns=columns, normalized=False)
        new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
        # new_dist = get_corrected_dist(true_proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
        new_dists.append(new_dist)

    else:
      err_keys, err_boundaries = list(zip(*sorted(err_ranges.dict.items())))
      if nondiff:
        nondiff_boundaries = list(set(err_boundaries))
        for nondiff_err_vals in itertools.product(*nondiff_boundaries):
          boundary_assn = dict(zip(nondiff_boundaries, nondiff_err_vals))
          assn = tuple(itertools.product(*[range(2) for _ in columns]))
          err_dist = {a: boundary_assn[err_ranges.dict[a]] for a in assn}
          err_dist = Distribution(data=err_dist, columns=columns, normalized=False)
          new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
          # new_dist = get_corrected_dist(true_proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
          new_dists.append(new_dist)

      else:
        if len(err_boundaries) > 16 and len(err_boundaries[0]) > 1:
          raise ValueError("{} is too many".format(2 ** len(err_boundaries)))
        for err_vals in itertools.product(*err_boundaries):
          err_dist = dict(zip(itertools.product(*[range(2) for _ in columns]), err_vals))
          err_dist = Distribution(data=err_dist, columns=columns, normalized=False)
          new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
          # new_dist = get_corrected_dist(true_proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
          new_dists.append(new_dist)

  if debug:
    print("constructed {} new dists".format(len(new_dists)))
    printout(true_dist, new_dists)


  # print("truth  ", {key: float("{:.2f}".format(val))
  #                   for key, val in true_dist.dict.items()})
  # print("proxy  ", {key: float("{:.2f}".format(val))
  #                   for key, val in proxy_dist.dict.items()})
  # print("corre  ", {key: float("{:.2f}".format(val))
  #                   for key, val in new_dists[0].dict.items()})
  # print("err    ", {key: float("{:.2f}".format(val))
  #                   for key, val in err_dist.dict.items()})
  recons_err = np.sum(np.absolute([true_dist.dict[key] - new_dists[0].dict[key]
                                  for key in true_dist.dict]))
  # print("recons err: {:.5f}".format(recons_err))

  extras = {'corrector_error': corrector_error, 'err_contained': err_contained}
  return (true_dist, new_dists, extras)


def corrector_oracle(sampler, corrector_error_width, n_train,
                     nondiff=False, debug=False):
  # We're just going to use "enough" data to get a very good estimate of the error,
  #   then "oracle-ize" it by degrading the accuracy of that estimate
  n_dev = min(int(1e5), 10 * n_train)
  error_offset_mean = corrector_error_width / 10

  true_dist = sampler.dist
  full_dim = len(sampler.dist.columns)
  proxy_var = 'u0'
  proxy_i = true_dist.columns.index(proxy_var)
  # proxy_dist, true_errs = construct_proxy_dist(true_dist, classifier_error, proxy_var, nondiff)

  # train the model on train data
  train_vars = np.array(sampler.sample_truth(n_train))
  train_text = sampler.sample_text(train_vars)

  train = np.concatenate((train_vars, train_text), axis=1)
  train_features = np.concatenate((train[:, :proxy_i], train[:, (1 + proxy_i):]), axis=1)
  model = sklearn.linear_model.LogisticRegression()
  model.fit(train_features, train[:, proxy_i])

  # infer proxy on dev data
  dev_vars = np.array(sampler.sample_truth(n_dev))
  dev_text = sampler.sample_text(dev_vars)
  dev = np.concatenate((dev_vars, dev_text), axis=1)
  dev_features = np.concatenate((dev[:, :proxy_i], dev[:, (1 + proxy_i):]),
                                axis=1)
  dev_proba = model.predict_proba(dev_features)[:, 0]
  dev_proxy = dev.copy().astype(np.float64)
  dev_proxy[:, proxy_i] = dev_proba

  true_err_ranges = calculate_error_matrix(
      dev_proxy[:, :full_dim], dev[:, :full_dim], proxy_var, true_dist.columns,
      nondiff=nondiff, debug=debug)

  # assume we had used enough dev data to perfectly estimate error
  # now what happens if we degrade our estimated error?
  err_ranges = {}
  for assn, true_err in true_err_ranges.dict.items():
    error_offset = np.random.normal(error_offset_mean, error_offset_mean)
    error_range_center = true_err[0] + error_offset * np.random.choice([-1, 1])
    err_ranges[assn] = [error_range_center - (corrector_error_width / 2),
                        error_range_center + (corrector_error_width / 2)]

  if debug:
    prec = 4
    print("true_err: ", {key: (round(val[0], prec), round(val[0], prec))
                         for key, val in true_err_ranges.items()})
    print("noisy_err:", {key: (round(val[0], prec), round(val[1], prec))
                         for key, val in err_ranges.items()})

  new_dists = []
  err_boundaries = [val for key, val in sorted(err_ranges.items())]
  err_keys = [key for key, _ in sorted(err_ranges.items())]
  proxy_dist = get_fractional_dist(dev_proxy, true_dist.columns, proxy_var)
  for err_vals in itertools.product(*err_boundaries):
    err_dist = dict(zip(err_keys, err_vals))
    err_dist = Distribution(data=err_dist, columns=proxy_dist.columns, normalized=False)
    new_dist = get_corrected_dist(proxy_dist, err_dist, proxy_var, truth=None, debug=debug)
    new_dists.append(new_dist)

  # print("constructed {} new dists".format(new_dists))
  if debug:
    printout(true_dist, new_dists)

  return (true_dist, new_dists)


def printout(test_dist, new_dists, score=gformula):

  corrected_effects = []
  for new_dist in new_dists:
    val = score(new_dist)
    if -1 < val < 1:
      corrected_effects.append(val)

  prec = 5
  if test_dist is not None:
    oracle_effect = score(test_dist)
    print("oracle: {:.{prec}f}".format(oracle_effect, prec=prec))

  print("corrected ({} of {} fit assumptions):".format(len(corrected_effects), len(new_dists)))
  table = OrderedDict()
  table['min'] = min(corrected_effects)
  table['p25'] = np.percentile(corrected_effects, 25)
  table['mean'] = np.mean(corrected_effects)
  table['p75'] = np.percentile(corrected_effects, 75)
  table['max'] = max(corrected_effects)

  width = 1 + max(
      max(len("{:.{prec}f}".format(x, prec=prec)) for x in table.values()),
      max(len(x) for x in table.keys()))
  print("".join(["{:^{width}s}".format(key, width=width) for key in table.keys()]))
  print("".join(["{:{width}.{prec}f}".format(
      val, width=width, prec=prec) for val in table.values()]))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("setup", type=str, help="double, classifier, corrector")
  parser.add_argument("--classifier_error", type=float,
                      default=0.0, help="what error should the classifier have")
  parser.add_argument("--corrector_error", type=float,
                      default=0.0, help="what error should the corrector classifier have")
  parser.add_argument("--logn_examples", type=float,
                      default=3, help="for trained model, how many (log10) examples?")
  parser.add_argument("--alpha", type=float, default=None,
                      help="alpha for conf int")
  parser.add_argument("--bootstrap", type=int, default=1,
                      help="How many bootstrap resamples?")
  parser.add_argument("--sample_err_rates", type=int, default=0,
                      help="How many samples to draw from err distributions?")
  parser.add_argument("--n_func", type=str, default=None,
                      help="transform for parametric intervals")
  parser.add_argument("--k", type=int, default=1, help="how many runs for each?")
  # parser.add_argument("--result_type", type=str, default="coverage")
  parser.add_argument("--debug", action='store_true', default=False)
  parser.add_argument("--write", type=str, default="append")
  parser.add_argument("--outdir", type=str, default="results/")
  parser.add_argument("--dist_seed", type=int, default=1)
  parser.add_argument("--exp_seed", type=int, default=1)
  parser.add_argument("--nondiff", action='store_true')
  parser.add_argument("--proxy_logits", action='store_true')
  parser.add_argument("--c_dim", type=int, default=1)
  parser.add_argument("--u_dim", type=int, default=1)
  parser.add_argument("--ay_effect", type=float, default=None)
  args = parser.parse_args()

  sampler = SyntheticData(
      ay_effect=args.ay_effect,
      seed=args.dist_seed, c_dim=args.c_dim, u_dim=args.u_dim)

  def get_outfn(args):
    setup = args.setup.lower()
    if setup == 'double':
      base = "double-{}-{}".format(
          args.classifier_error, args.corrector_error)
    elif setup == 'double_noint':
      base = "double-noint-{}-{}".format(
          args.classifier_error, args.corrector_error)
    elif setup == 'classifier':
      base = "classifier-{}-{}".format(
          args.classifier_error, args.logn_examples)
    elif setup == 'uncorrected':
      base = "uncorrected-{}".format(args.classifier_error)
    elif setup == 'corrector':
      base = "corrector-{}-{}".format(
          args.corrector_error, args.logn_examples)
    else:
      raise ValueError("Unknown setup: '{}'".format(args.setup))

    seeds = "-{}-{}-{}".format(args.dist_seed, args.exp_seed, args.k)

    return "{}{}.json".format(base, seeds)

  interval_widths = list(range(20, 100, 10)) + [95]

  if args.n_func is not None:
    if args.n_func.startswith("n**2x:"):
      x = float(args.n_func[6:])
      def n_func(n):
        return n ** (2 * x)
    else:
      raise ValueError("Unknown n_func")
  else:
    n_func = None

  def run(args, seed):
    n_examples = int(10 ** args.logn_examples)
    setup = args.setup.lower()
    if setup == 'double':
      if args.alpha is not None:
        raise ValueError("alpha must be none for double oracle")
      truth, estimates = double_oracle(
          sampler, args.classifier_error, args.corrector_error,
          nondiff=args.nondiff, debug=args.debug)
      return get_results(truth, estimates)
    elif setup == 'double_noint':
      if args.alpha is not None:
        raise ValueError("alpha must be none for double oracle, no int")
      truth, estimates = double_oracle_no_interval(
          sampler, args.classifier_error, args.corrector_error,
          nondiff=args.nondiff, debug=args.debug)
      return get_results(truth, estimates)
    elif setup == 'classifier':
      truth, estimates, extras = classifier_oracle(
          sampler, args.classifier_error, n_examples,
          proxy_logits=args.proxy_logits,
          sample_err_rates=args.sample_err_rates,
          n_func=n_func,
          nondiff=args.nondiff, bootstrap=args.bootstrap,
          alpha=args.alpha, debug=args.debug, seed=seed)
      return get_results(truth, estimates, extras=extras, interval_widths=interval_widths)
    elif setup == 'uncorrected':
      truth, estimates = uncorrected_classifier(
          sampler, args.classifier_error,
          nondiff=args.nondiff, debug=args.debug)
      return get_results(truth, estimates)
    elif setup == 'corrector':
      if args.alpha is not None:
        raise ValueError("alpha must be none for corrector oracle")
      truth, estimates = corrector_oracle(
          sampler, args.corrector_error, n_examples,
          nondiff=args.nondiff, debug=args.debug)
      return get_results(truth, estimates, interval_widths=interval_widths)
    else:
      raise ValueError("Unknown setup: '{}'".format(args.setup))

  # having seeded data dist, seed experiments
  np.random.seed(args.exp_seed)
  iinfo = np.iinfo(np.int32)
  exp_seeds = np.random.randint(0, iinfo.max, size=args.k)
  raw_results = []
  errors = defaultdict(list)
  for seed in exp_seeds:
    try:
      raw_results.append(run(args, seed))
    except Exception as e:
      errors[type(e)].append(str(e))
      if args.debug:
        raise e
      pass
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
  print("min/p2.5/mean/p97.5/max: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(
      means.get('min', np.nan),
      means.get('p2.5', np.nan), means.get('mean', np.nan),
      means.get('p97.5', np.nan), means.get('max', np.nan)))
  if args.debug and args.setup == 'classifier':
    for width in interval_widths:
      print("at {}%, coverage: {:.3f} width: {:.3f}".format(
          width, means['trunc{}cov'.format(width)], means['trunc{}width'.format(width)]))
    print("at 100%, coverage: {:.3f} width: {:.3f}".format(
        means['covered'], means['width']))

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


if __name__ == "__main__":
  main()
