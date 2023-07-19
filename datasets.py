# Code from causal_sensitivity_2020 repository by @zachwooddoughty

# import pdb; pdb.set_trace()

import os
import gzip
import json

import itertools
import numpy as np
import scipy.stats

from numpy.random import beta, binomial, choice, uniform
from utils import Distribution, gformula


synthetic_config = {
    'topic_std': 0.2,
    'missing_bias': 0.7,
    'missing_effect_std': 0.1,
    'missing_effects': (.2, -.4),
    'vocab_size': 4334,
    'c_bias': 0.1,
    'a_bias': 0.1,
    'ca_effect': 0.5,
    'y_bias': 0.2,
    'cy_effect': 0.3,
    'ay_effect': -0.3,
}


class SyntheticData:
  def __init__(self, c_dim=1, u_dim=1, ay_effect=None, seed=None, **kwargs):
    if ay_effect is None:
      self.dist = UnrestrictedDist(c_dim=c_dim, u_dim=u_dim, seed=seed)
    else:
      self.dist = SpecifiedEffectDist(
          c_dim=c_dim, u_dim=u_dim, ay_effect=ay_effect, seed=seed)

    full_dim = 2 + c_dim + u_dim

    valid_args = ['vocab_size', 'topic_std', 'topic_bias', 'nondiff_text']
    for arg in kwargs.keys():
      assert arg in valid_args, arg

    self.nondiff_text = kwargs.get('nondiff_text', None)
    self.vocab_size = kwargs.get('vocab_size', 4334)
    self.topic_std = kwargs.get('topic_std', 0.2)
    self.topic_bias = np.ones(self.vocab_size) * kwargs.get('topic_bias', 0.2)
    self.topic_effects = self.get_topic_effects(full_dim, seed)

  def get_topic_effects(self, full_dim, seed=None):
    if seed is not None:
      np.random.seed(seed)

    topic_effects = [np.random.choice([-1., 1.], self.vocab_size) *
                     np.random.normal(0, self.topic_std, self.vocab_size)
                     for _ in range(full_dim)]
    num_nonzero_effects = max(100, self.vocab_size // 50)
    topic_effects_mask = np.concatenate(
        [np.ones(num_nonzero_effects),
         np.zeros(self.vocab_size - num_nonzero_effects)])
    np.random.shuffle(topic_effects_mask)
    topic_effects *= topic_effects_mask
    #specify specific variables
    if self.nondiff_text is not None:
      var_list = self.nondiff_text.split(',')
      var_list.append('u0')
      for i, var in enumerate(self.dist.columns):
        if var not in var_list:
          topic_effects[i, :] = np.zeros(self.vocab_size)
    return topic_effects

  def true_dist(self, round_to=None):
    return self.dist.true_dist(round_to)

  def sample_truth(self, n=1):
    raw_dist = self.dist.dict
    assns = sorted(raw_dist.keys())
    dist_cumsum = np.cumsum([raw_dist[assn] for assn in assns])

    def sample():
      x = np.random.uniform(0, 1)
      for i, val in enumerate(dist_cumsum):
        if x < val:
          return assns[i]

    return np.array([sample() for _ in range(n)])

  def sample_text(self, truth):
    ''' Sample text variables from the (a, y, c, u) truth variables '''
    n, full_dim = truth.shape
    topic = np.tile(self.topic_bias, n).reshape(n, self.vocab_size)
    for i in range(full_dim):
      topic += truth[:, i].reshape(n, 1) * np.tile(
          self.topic_effects[i], n).reshape(n, self.vocab_size)
    topic = np.clip(topic, 0.01, 0.99)
    words = []
    for i in range(n):
      word = (np.random.random(self.vocab_size) < topic[i]).astype(np.int32)
      words.append(word)
    words = np.array(words)
    return words


class SpecifiedEffectDist(Distribution):
  def __init__(self, ay_effect=0.05, c_dim=1, u_dim=1, seed=None):
    if seed is not None:
      np.random.seed(seed)
    self.c_dim = c_dim
    self.u_dim = u_dim
    self.ay_effect = ay_effect
    self.full_dim = 2 + c_dim + u_dim
    ay_cols = ['a', 'y']
    c_cols = ["c{}".format(i) for i in range(c_dim)]
    u_cols = ["u{}".format(i) for i in range(u_dim)]
    self.columns = ay_cols + c_cols + u_cols

    acu_vals = np.random.random(size=2 ** (self.full_dim - 1))
    acu_vals /= np.sum(acu_vals)
    acu_vars = ['a'] + c_cols + u_cols
    acu_dist = dict(zip(itertools.product(*[range(2) for _ in acu_vars]),
                        acu_vals))
    acu_dist = Distribution(data=acu_dist, columns=acu_vars)
    cu_vars = c_cols + u_cols

    ay_dim = 1  # constant 4 is the dimensionality of p(a, y)
    y1_mean_dist = np.random.random(size=2 ** (self.full_dim - 2))
    y1_mean_dist = y1_mean_dist * (1 - abs(ay_effect)) + abs(ay_dim * ay_effect) / 2
    assert(np.all(y1_mean_dist > 0))

    y1_a0_dist = y1_mean_dist - (ay_dim * ay_effect / 2)
    y1_a1_dist = y1_mean_dist + (ay_dim * ay_effect / 2)

    y1_a0_dist = dict(zip(itertools.product(*[range(2) for _ in cu_vars]), y1_a0_dist))
    y1_a1_dist = dict(zip(itertools.product(*[range(2) for _ in cu_vars]), y1_a1_dist))
    y1_a0_dist = Distribution(data=y1_a0_dist, columns=cu_vars, normalized=False)
    y1_a1_dist = Distribution(data=y1_a1_dist, columns=cu_vars, normalized=False)
    
    full_dist = {}
    assns = itertools.product(*[range(2) for _ in range(self.full_dim)])
    for assn in assns:
      mapping = dict(zip(self.columns, assn))
      acu_prob = acu_dist.get(**{var: mapping[var] for var in acu_vars})
      if mapping['a'] == 0:
        y_prob = y1_a0_dist.get(**{var: mapping[var] for var in cu_vars})
      else:
        y_prob = y1_a1_dist.get(**{var: mapping[var] for var in cu_vars})
      if mapping['y'] == 0:
        y_prob = 1 - y_prob

      full_dist[tuple(assn)] = y_prob * acu_prob

    assert(np.all(np.array(list(full_dist.values())) > 0))
    super().__init__(data=full_dist, columns=self.columns)

  def true_dist(self, round_to=None):
    dist = {}
    assns = itertools.product(*[range(2) for _ in range(self.full_dim)])
    for assn, prob in zip(assns, self.dist.dict):
      if round_to is not None:
        prob = round(prob, round_to)

      dist[tuple(assn)] = prob

    return dist


class UnrestrictedDist(Distribution):
  def __init__(self, c_dim=1, u_dim=1, seed=None):
    if seed is not None:
      np.random.seed(seed)
    self.c_dim = c_dim
    self.u_dim = u_dim
    self.full_dim = 2 + c_dim + u_dim
    ay_cols = ['a', 'y']
    c_cols = ["c{}".format(i) for i in range(c_dim)]
    u_cols = ["u{}".format(i) for i in range(u_dim)]
    self.columns = ay_cols + c_cols + u_cols

    dist = np.random.random(size=2 ** self.full_dim)
    dist = dist / np.sum(dist)
    assns = itertools.product(*[range(2) for _ in range(self.full_dim)])
    full_dist = dict(zip(assns, dist))
    super().__init__(data=full_dist, columns=self.columns)


def test_distribution_gformula():
  for c_dim in [2, 3]:
    for i in range(10):
      ay_effect = np.random.random() * 2 - 1
      print("c_dim is {}, effect is: {:.3}".format(c_dim, ay_effect))
      se = SpecifiedEffectDist(ay_effect=ay_effect, seed=i, c_dim=3)
      est = gformula(se)
      # print(np.round(ay_effect / est, 3), end=', ')
      assert np.isclose(est, ay_effect)
    print()


if __name__ == "__main__":
  sd = SyntheticData(ay_effect=0.1, nondiff_text='u0,c0')
  test_distribution_gformula()
