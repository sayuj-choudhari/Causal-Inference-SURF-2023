import itertools
import math
import logging
import os
import json
import argparse
import random
from collections import OrderedDict, defaultdict
from statistics import mean
from statistics import variance

import numpy as np
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

from datasets import SyntheticData
from utils import Distribution, gformula, NumpySerializer
from utils import get_dist, get_fractional_dist
from utils import construct_proxy_dist
from utils import construct_model_proxy_dist
from sensitivity import clopper_pearson_interval

from line_profiler import LineProfiler

import matplotlib.pyplot as plt

import warnings
# warnings.simplefilter("error")
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def read_json(json_dir):
    with open(json_dir, "r") as json_file:
        data = json.load(json_file)

    return data["mean_abs_abs_mean"], data['std_max']

def get_result_dir(cdim, logn, method_type, k, model_type):
  if method_type == 'model':
    base = "{}_{}_experiment_{}".format(model_type, method_type, logn)
  else:
    base = "{}_experiment_{}".format(method_type, logn)
  seeds = "-{}-{}".format(cdim, k)
  return "{}{}.json".format(base, seeds)

def random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)

def plotter(x_data, y_data, y_std, method_type, model_type):
    plot_color = random_color()
    if model_type is not None:
        plt.plot(x_data, y_data, label = model_type + ' ' + method_type + ' data', color = plot_color)
    else:
        plt.plot(x_data, y_data, label = method_type + ' data', color = plot_color)

    y_list = [y_data[i] + y_std[i] for i in range(len(y_data))]
    plt.plot(x_data, [y_data[i] + y_std[i] for i in range(len(y_data))], color = plot_color)

def create_plot_data(cdim, logn, method_type, k, lower_cdim = None, upper_cdim = None, cdim_step = None,
             lower_logn = None, upper_logn = None, logn_step = None, model_type = None):
    if cdim_step is None:
        assert(logn_step is not None)
        causal_error = []
        causal_std = []
        logn = []
        for i in range(lower_logn, upper_logn + 1, logn_step):
           logn.append(i)
           json_dir = 'experiments/' + get_result_dir(cdim, float(i), method_type, k, model_type)
           if not os.path.exists(json_dir):
              if model_type is not None:
                os.system("python measurement_error.py --logn_examples {} --c_dim {}  --method_type {} --model_type {} --k {}".format(i, cdim, method_type, model_type, k))
              else:

                os.system("python measurement_error.py --logn_examples {} --c_dim {}  --method_type {} --k {}".format(i, cdim, method_type, k))     
            
           
           mean_error, std = read_json(json_dir)
           causal_error.append(mean_error)
           causal_std.append(std)

        return causal_error, causal_std, logn
    
    elif logn_step is None:
        assert(cdim_step is not None)
        causal_error = []
        causal_std = []
        cdim = []
        for i in range(lower_cdim, upper_cdim + 1, cdim_step):
           cdim.append(i)
           json_dir = 'experiments/' + get_result_dir(i, float(logn), method_type, k, model_type)
           if not os.path.exists(json_dir):
              if model_type is not None:
                os.system("python measurement_error.py --logn_examples {} --c_dim {}  --method_type {} --model_type {} --k {}".format(logn, i, method_type, model_type, k))
              else:
                os.system("python measurement_error.py --logn_examples {} --c_dim {}  --method_type {} --k {}".format(logn, i, method_type, k))     
           
           mean_error, std = read_json(json_dir)
           causal_error.append(mean_error)
           causal_std.append(abs(std))

        return causal_error, causal_std, cdim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdim", type=int, default = 4, help="how many observed confounders")

    parser.add_argument("--logn", type = float, default = 4, help="how many examples (log10)")

    parser.add_argument("--k", type=int, default=3, help="how many runs for each?")
    parser.add_argument("--method_type", type=str, default='model')
    parser.add_argument("--matrix", type =  bool, default = False)
    parser.add_argument("--model", type = bool, default = False)
    parser.add_argument("--regression", type = bool, default = False)
    parser.add_argument("--perfect", type = bool, default = False)
        
    parser.add_argument("--upper_cdim", type=int, default=None)
    parser.add_argument("--lower_cdim", type=int, default=None)
    parser.add_argument("--cdim_step", type=int, default=None)

    parser.add_argument("--upper_logn", type=int, default=None)
    parser.add_argument("--lower_logn", type=int, default=None)
    parser.add_argument("--logn_step", type=int, default=None) 

    args = parser.parse_args()

    if args.matrix:
        y_data, y_std, x_data = create_plot_data(args.cdim, args.logn, 'matrix', args.k, args.lower_cdim,
                                                args.upper_cdim, args.cdim_step, args.lower_logn,
                                                args.upper_logn, args.logn_step, None)
        
        plotter(x_data, y_data, y_std, 'matrix', None)

    if args.model:
       model_list = [args.regression, args.perfect]
       if model_list[0]:
        y_data, y_std, x_data = create_plot_data(args.cdim, args.logn, 'model', args.k, args.lower_cdim,
                                                args.upper_cdim, args.cdim_step, args.lower_logn,
                                                args.upper_logn, args.logn_step, 'regression')
        
        plotter(x_data, y_data, y_std, 'model', 'regression')
       if model_list[1]:
        y_data, y_std, x_data = create_plot_data(args.cdim, args.logn, 'model', args.k, args.lower_cdim,
                                                args.upper_cdim, args.cdim_step, args.lower_logn,
                                                args.upper_logn, args.logn_step, 'perfect')
        
        plotter(x_data, y_data, y_std, 'model', 'perfect')

    plt.legend()

    if args.cdim_step is None:
       plt.xlabel('logn examples')
       plt.title('Causal error estimate vs. logn examples of data')
    if args.logn_step is None:
       plt.xlabel('Amount of observed confounders influencing proxy classification')
       plt.title('Causal error estimate vs. dimensionality of observed confounding')

    plt.ylabel('Overall causal error')
    plt.show()

if __name__ == "__main__":
   main()


    

    


