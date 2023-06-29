# Code from causal_sensitivity_2020 repository by @zachwooddoughty

import numpy as np
import scipy.stats

def clopper_pearson_interval(n, p, alpha, return_triple=False):
  '''
  https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
  n: number of trials
  p: proportion of successes
  alpha: confidence interval param (e.g. .95 for 95% conf interval)
  '''

  # n = np.sqrt(n)

  low_q = .5 - alpha / 2
  high_q = .5 + alpha / 2

  if np.isnan(p) or p <= 0:
    low = 0
  else:
    low = scipy.stats.beta.ppf(low_q, p*n, n - p*n + 1)

  if np.isnan(p) or p >= 1:
    high = 1
  else:
    high = scipy.stats.beta.ppf(high_q, p*n + 1, n - p*n)

  if return_triple:
    return (low, p, high)
  else:
    return (low, high)


# invert to solve for alpha to get point estimate
# alpha = 2 * (.5 - scipy.stats.beta.cdf(.5, p*n, n - p*n + 1))
# alpha = 1 gives a meaningless maximal interval,
#   but alpha = (1 - 1e-10) still gives a very narrow interval!
