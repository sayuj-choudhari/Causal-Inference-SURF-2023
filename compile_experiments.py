import os, json
import re
import itertools
import glob
from collections import defaultdict

import numpy as np


def default_args():
  return dict(
    classifier_err=[0.3],
    k=[100],
    exp_seed=[1],
    dist_seed=[1],
    logn_examples=[3.0],
  )


def classifier_fn(**kwargs):
  return "classifier-{}-{}-{}-{}-{}.json".format(
      kwargs['classifier_err'], kwargs['logn_examples'],
      kwargs['dist_seed'], kwargs['exp_seed'], kwargs['k'])


def uncorrected_fn(**kwargs):
  return "uncorrected-{}-{}-{}-{}.json".format(
      kwargs['classifier_err'],
      kwargs['dist_seed'], kwargs['exp_seed'], kwargs['k'])

def me_fn(**kwargs):
  return "me_{}-{}-{}-{}.json".format(
      kwargs['logn_examples'],
      kwargs['dist_seed'], kwargs['exp_seed'], kwargs['k'])


def reader(results_dir, fn_func, **kwargs):

  keys, vals = list(zip(*kwargs.items()))
  for val in vals:
    assert type(val) in (tuple, list)

  for val_option in itertools.product(*vals):
    assn = dict(zip(keys, val_option))
    fn = fn_func(**assn)
    infn = os.path.join(results_dir, fn)
    if not os.path.exists(infn):
      print("can't find {}".format(infn))
      continue
    with open(infn) as inf:
      for line in inf:
        obj = json.loads(line)
        yield (infn, obj)


def compile_sample_transform():
  results_dir = os.path.join('experiments', 'interval_compare2', 'json')
  fixed = ['logn_examples', 'c_dim', 'classifier_error', 'nondiff',
           'sample_err_rates']
  independent = ['n_func']

  classifier_errs = [0.1, 0.3]
  # logn_examples = [x / 10 for x in range(10, 55, 5)]
  logn_examples = [3.0]
  dist_seed = list(range(101, 111))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
    sample_err_rates=[100],
  ))

  results = defaultdict(lambda: defaultdict(list))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])

    if obj.get('alpha') is not None or obj.get('bootstrap') > 1:
      continue
    if obj.get('n_func') is not None:
      key, val = obj['n_func'].split(':')
      if key == 'n**2x':
        ind_key = val
        results[fixed_key][ind_key].append((
              obj['mean_trunc95cov'],
              obj['mean_trunc95width']))

  headers = fixed + independent
  headers.extend(["cov_mean", "cov_std", "width_mean", "width_std"])

  for fixed_key in results:
    outfn = os.path.join(
        results_dir, "{}-sample-transform.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      outf.write("{}\n".format(" ".join(headers)))
      lines = []
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')
        covs, widths = zip(*results[fixed_key][ind_key])
        cov_mean = "{}".format(np.mean(covs))
        cov_std = "{}".format(np.std(covs))
        width_mean = "{}".format(np.mean(widths))
        width_std = "{}".format(np.std(widths))

        vals = fixed_vals + ind_vals + [cov_mean, cov_std, width_mean, width_std]
        line = "{}\n".format(" ".join(vals))
        lines.append((cov_mean, line))

      for _, line in sorted(lines):
        outf.write(line)

def compile_alpha_transform():
  results_dir = os.path.join('experiments', 'interval_compare2', 'json2')
  fixed = ['logn_examples', 'c_dim', 'classifier_error', 'nondiff', 'alpha']
  independent = ['n_func']

  classifier_errs = [0.1, 0.3]
  # logn_examples = [x / 10 for x in range(10, 55, 5)]
  logn_examples = [3.0]
  dist_seed = list(range(101, 111))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
    alpha=[0.95],
  ))

  results = defaultdict(lambda: defaultdict(list))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])

    if obj.get('sample_err_rates') > 0 or obj.get('bootstrap') > 1:
      continue
    if obj.get('n_func') is not None:
      key, val = obj['n_func'].split(':')
      if key == 'n**2x':
        ind_key = val
        results[fixed_key][ind_key].append((
              obj['mean_covered'],
              obj['mean_width']))

  headers = fixed + independent
  headers.extend(["cov_mean", "cov_std", "width_mean", "width_std"])

  for fixed_key in results:
    outfn = os.path.join(results_dir, "{}-alpha-transform.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      outf.write("{}\n".format(" ".join(headers)))
      lines = []
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')
        covs, widths = zip(*results[fixed_key][ind_key])
        cov_mean = "{}".format(np.mean(covs))
        cov_std = "{}".format(np.std(covs))
        width_mean = "{}".format(np.mean(widths))
        width_std = "{}".format(np.std(widths))

        vals = fixed_vals + ind_vals + [cov_mean, cov_std, width_mean, width_std]
        line = "{}\n".format(" ".join(vals))
        lines.append((cov_mean, line))

      # sort by coverage
      for _, line in sorted(lines):
        outf.write(line)


def compile_alpha_interval_compare():
  results_dir = os.path.join('experiments', 'interval_compare2', 'json')
  fixed = ['logn_examples', 'c_dim', 'classifier_error', 'nondiff']
  independent = ['alpha']

  classifier_errs = [0.1, 0.3]
  # logn_examples = [x / 10 for x in range(10, 55, 5)]
  logn_examples = [3.0]
  dist_seed = list(range(101, 111))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
  ))

  results = defaultdict(lambda: defaultdict(list))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    ind_key = "--".join(["{}".format(obj[x]) for x in independent])

    if obj['alpha'] is not None:
      results[fixed_key][ind_key].append((
            obj['mean_covered'],
            obj['mean_width']))

  headers = fixed + independent
  headers.extend(["cov_mean", "cov_std", "width_mean", "width_std"])

  for fixed_key in results:
    outfn = os.path.join(results_dir, "{}-alpha-sweep.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      outf.write("{}\n".format(" ".join(headers)))
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')
        covs, widths = zip(*results[fixed_key][ind_key])
        cov_mean = "{}".format(np.mean(covs))
        cov_std = "{}".format(np.std(covs))
        width_mean = "{}".format(np.mean(widths))
        width_std = "{}".format(np.std(widths))

        vals = fixed_vals + ind_vals + [cov_mean, cov_std, width_mean, width_std]
        outf.write("{}\n".format(" ".join(vals)))
        print(" ".join(vals))


def compile_interval_compare():
  results_dir = os.path.join('experiments', 'interval_compare2', 'json')
  fixed = ['logn_examples', 'c_dim', 'classifier_error', 'nondiff']
  # independent = ['bootstrap', 'sample_err_rates']
  independent = ['bootstrap', 'sample_err_rates', 'alpha']

  classifier_errs = [0.1, 0.3]
  # logn_examples = [x / 10 for x in range(10, 55, 5)]
  logn_examples = [3.0]
  dist_seed = list(range(101, 111))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
  ))

  interval_widths = list(range(20, 100, 10))
  results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    ind_key = "--".join(["{}".format(obj[x]) for x in independent])

    for w in interval_widths:
      results[fixed_key][ind_key][w].append((
          obj['mean_trunc{}cov'.format(w)],
          obj['mean_trunc{}width'.format(w)]))
    results[fixed_key][ind_key][100].append((
          obj['mean_covered'],
          obj['mean_width']))

  output_widths = interval_widths + [100]
  headers = fixed + independent
  headers.extend(["trunc", "cov_mean", "cov_std", "width_mean", "width_std"])
  #   ["{}{}".format(w, output_header) for w in output_widths
  #    for output_header in ["cov_mean", "cov_std", "width_mean", "width_std"]])

  # print("wids "+ " ".join(["{:5d}".format(x) for x in output_widths]))
  for fixed_key in results:
    for ind_key in sorted(results[fixed_key]):
      outfn = os.path.join(results_dir, "{}-{}.dat".format(fixed_key, ind_key))
      with open(outfn, "w") as outf:
        outf.write("{}\n".format(" ".join(headers)))
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')
        # print(ind_key)
        all_covs = []
        all_widths = []
        for w in output_widths:
          covs, widths = zip(*results[fixed_key][ind_key][w])
          width = "{}".format(w)
          cov_mean = "{}".format(np.mean(covs))
          cov_std = "{}".format(np.std(covs))
          width_mean = "{}".format(np.mean(widths))
          width_std = "{}".format(np.std(widths))

          vals = fixed_vals + ind_vals + [
              width, cov_mean, cov_std, width_mean, width_std]
          outf.write("{}\n".format(" ".join(vals)))
          print(" ".join(vals))

      # print("covs "+ " ".join(["{:5.3f}".format(x) for x in all_covs]))
      # print("wids "+ " ".join(["{:5.3f}".format(x) for x in all_widths]))


def need_sensitivity_classifier():
  results_dir = os.path.join('experiments', 'need_sensitivity2', 'json')
  fixed = ['classifier_error', 'c_dim']
  independent = ['logn_examples',]

  classifier_errs = [0.1, 0.3]
  logn_examples = [x / 10 for x in range(10, 55, 5)]
  dist_seed = list(range(1, 11))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
  ))

  results = defaultdict(lambda: defaultdict(list))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    ind_key = "--".join(["{}".format(obj[x]) for x in independent])

    try:
      results[fixed_key][ind_key].append(obj['mean_abs_mean'])
    except Exception as e:
      print("failed on {} with {}".format(infn, repr(e)))

  for fixed_key in results:
    outfn = os.path.join(results_dir, "ns_{}.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      headers = fixed + independent + ['mean', 'std']
      outf.write("{}\n".format(" ".join(headers)))
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')
        mean = "{}".format(np.mean(results[fixed_key][ind_key]))
        std = "{}".format(np.std(results[fixed_key][ind_key]))

        vals = fixed_vals + ind_vals + [mean, std]
        outf.write("{}\n".format(" ".join(vals)))
        print(fixed_key, ind_key, mean, std)

  # Now do uncorrected dat (which doesn't vary with logn)
  uncorrected_args = default_args()
  uncorrected_args.update(dict(
    classifier_err=classifier_errs,
    dist_seed=dist_seed,
  ))

  results = defaultdict(list)
  for infn, obj in reader(results_dir, uncorrected_fn, **uncorrected_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    try:
      results[fixed_key].append(obj['mean_abs_mean'])
    except Exception as e:
      print("failed on {} with {}: {}".format(infn, type(e), str(e)))

  for fixed_key in results:
    outfn = os.path.join(results_dir, "unc_{}.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      headers = fixed + independent + ['mean', 'std']
      outf.write("{}\n".format(" ".join(headers)))
      for ind_vals in itertools.product(
          *[classifier_args[key] for key in independent]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ["{}".format(x) for x in ind_vals]
        mean = "{}".format(np.mean(results[fixed_key]))
        std = "{}".format(np.std(results[fixed_key]))

        vals = fixed_vals + ind_vals + [mean, std]
        outf.write("{}\n".format(" ".join(vals)))
      print(fixed_key, mean, std)


def compile_sensitivity_helps_oracle():
  results_dir = os.path.join('experiments', 'sensitivity_helps2', 'json2')
  fixed = ['c_dim', 'classifier_error', 'nondiff',
           'alpha', 'bootstrap', 'sample_err_rates', 'n_func']
  independent = ['logn_examples']

  classifier_errs = [0.1, 0.3]
  logn_examples = [x / 10 for x in range(10, 55, 5)]
  # logn_examples = [x / 10 for x in range(10, 35, 5)]
  # logn_examples = [3.0]
  dist_seed = list(range(1, 11))

  classifier_args = default_args()
  classifier_args.update(dict(
    classifier_err=classifier_errs,
    logn_examples=logn_examples,
    dist_seed=dist_seed,
  ))

  results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  for infn, obj in reader(results_dir, classifier_fn, **classifier_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    ind_key = "--".join(["{}".format(obj[x]) for x in independent])

    try:
      # if obj['alpha'] is None:
      #   low = obj['mean_p2.5']
      #   high = obj['mean_p97.5']
      # else:
      # low = obj['mean_p5.0']
      # high = obj['mean_p95.0']
      # cov90 = obj['mean_trunc90cov']
      low = obj['mean_min']
      high = obj['mean_max']
      cov90 = obj['mean_covered']

      results[fixed_key][ind_key]['low'].append(low)
      results[fixed_key][ind_key]['high'].append(high)
      results[fixed_key][ind_key]['cov'].append(cov90)

    except Exception as e:
      print("failed on {} with {}".format(infn, repr(e)))

  headers = fixed + independent + [
      'low_mean', 'low_std', 'high_mean', 'high_std', 'mean_cov']

  for fixed_key in results:
    outfn = os.path.join(results_dir, "class-{}.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      outf.write("{}\n".format(" ".join(headers)))
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')

        low_vals = results[fixed_key][ind_key]['low']
        low_mean = "{}".format(np.mean(low_vals))
        low_std = "{}".format(np.std(low_vals))
        high_vals = results[fixed_key][ind_key]['high']
        high_mean = "{}".format(np.mean(high_vals))
        high_std = "{}".format(np.std(high_vals))
        mean_cov = "{}".format(np.mean(results[fixed_key][ind_key]['cov']))

        vals = fixed_vals + ind_vals
        vals.extend([low_mean, low_std, high_mean, high_std, mean_cov])
        outf.write("{}\n".format(" ".join(vals)))


def compile_sensitivity_helps_me():
  results_dir = os.path.join('experiments', 'sensitivity_helps2', 'json2')
  fixed = ['c_dim', 'nondiff',
           'alpha', 'bootstrap', 'sample_err_rates']
  independent = ['logn_examples']

  k = [100]
  logn_examples = [x / 10 for x in range(20, 55, 5)]
  # logn_examples = [x / 10 for x in range(10, 35, 5)]
  # logn_examples = [3.0]
  dist_seed = list(range(1, 11))

  me_args = default_args()
  me_args.update(dict(
    logn_examples=logn_examples,
    dist_seed=dist_seed,
    k=k,
  ))

  results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  for infn, obj in reader(results_dir, me_fn, **me_args):
    fixed_key = "--".join(["{}".format(obj[x]) for x in fixed])
    ind_key = "--".join(["{}".format(obj[x]) for x in independent])

    try:
      low = obj['mean_p5.0']
      high = obj['mean_p95.0']
      cov90 = obj['mean_trunc90cov']

      results[fixed_key][ind_key]['low'].append(low)
      results[fixed_key][ind_key]['high'].append(high)
      results[fixed_key][ind_key]['cov'].append(cov90)

    except Exception as e:
      print("failed on {} with {}".format(infn, repr(e)))

  headers = fixed + independent + [
      'low_mean', 'low_std', 'high_mean', 'high_std', 'mean_cov', 'count']

  for fixed_key in results:
    outfn = os.path.join(results_dir, "me-{}.dat".format(fixed_key))
    with open(outfn, "w") as outf:
      outf.write("{}\n".format(" ".join(headers)))
      for ind_key in sorted(results[fixed_key]):
        fixed_vals = fixed_key.split('--')
        ind_vals = ind_key.split('--')

        low_vals = results[fixed_key][ind_key]['low']
        low_mean = "{}".format(np.mean(low_vals))
        low_std = "{}".format(np.std(low_vals))
        high_vals = results[fixed_key][ind_key]['high']
        high_mean = "{}".format(np.mean(high_vals))
        high_std = "{}".format(np.std(high_vals))
        mean_cov = "{}".format(np.mean(results[fixed_key][ind_key]['cov']))
        count = "{}".format(sum(1 for _ in low_vals))

        vals = fixed_vals + ind_vals
        vals.extend([low_mean, low_std, high_mean, high_std, mean_cov, count])
        outf.write("{}\n".format(" ".join(vals)))

if __name__ == "__main__":
  # need_sensitivity_classifier()

  # compile_interval_compare()
  # compile_alpha_interval_compare()
  compile_alpha_transform()
  # compile_sample_transform()

  # compile_sensitivity_helps_oracle()
  # compile_sensitivity_helps_me()
