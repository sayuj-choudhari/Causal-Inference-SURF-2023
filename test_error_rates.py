import numpy as np

from datasets import SyntheticData
from measurement_error import Experimental_Data
from measurement_error import calculate_error_matrix, correct
from measurement_error import get_fractional_error_rate, get_error_rate
from utils import construct_proxy_dist, get_fractional_dist, get_dist


def experiment(fractional=False, logn_examples=2, seed=1):
    proxy_var = "u0"
    mean = 0.01
    var = 0.005
    fractional_logit = 0.8

    np.random.seed(seed)
    dataset = SyntheticData()
    proxy_i = dataset.dist.columns.index(proxy_var)
    exp_data = Experimental_Data()
    _, true_errs = construct_proxy_dist(
        dataset.dist, mean, proxy_var, var, nondiff=False)

    n_examples = np.power(10., logn_examples)
    truth = dataset.sample_truth(int(n_examples)).astype(np.float64)
    proxy_arr = truth.copy()

    for row_i in range(truth.shape[0]):
        assn = dict(zip(dataset.dist.columns, truth[row_i, :].tolist()))
        err_rate = np.clip(true_errs.get(**assn), 0.01, 0.99)

        # If logit is 0.9, convert 0 to 0.1 and 1 to 0.9
        # If logit is 0.8, convert 0 to 0.2 and 1 to 0.8
        if fractional:
            low_prob = 1 - fractional_logit
            delta_prob = 2 * fractional_logit - 1
            proxy_arr[row_i, proxy_i] = low_prob + delta_prob * proxy_arr[row_i, proxy_i]

        # Introduce error with `err_rate` probability
        if np.random.binomial(1, err_rate):
            proxy_arr[row_i, proxy_i] = 1 - proxy_arr[row_i, proxy_i]

    error_matrix = calculate_error_matrix(
          exp_data, proxy_arr, truth, proxy_var, dataset.dist.columns)

    # How far off are our error estimates from the true error rates?
    total_err_err = np.sum(
        [np.abs(true_errs.dict[key] - error_matrix.dict[key])
         for key in error_matrix.dict.keys()])

    if fractional:
        new_dist = get_fractional_dist(proxy_arr, dataset.dist.columns, proxy_var)
    else:
        new_dist = get_dist(proxy_arr, dataset.dist.columns)

    new_dist = correct(new_dist, proxy_var, error_matrix)[0]
    total_dist_err = np.sum(
        [np.abs(dataset.dist.dict[key] - new_dist.dict[key])
         for key in new_dist.dict.keys()])

    return {"total_err_err": total_err_err, "total_dist_err": total_dist_err}


def main():
    logn_examples = [2, 3, 4]
    seeds = range(1, 20)
    agg = np.mean

    # Run experiments
    results = {True: {}, False: {}}
    for fractional in [True, False]:
        results[fractional] = {seed: {} for seed in seeds}
        for logn in logn_examples:
            results[fractional][logn] = []
            for seed in seeds:
                result = experiment(seed=seed, fractional=fractional, logn_examples=logn)
                results[fractional][logn].append(result)

    # Average over dicts
    for fractional in [True, False]:
        for logn in logn_examples:
            cell = results[fractional][logn]
            total_err_err = agg([x["total_err_err"] for x in cell])
            total_dist_err = agg([x["total_dist_err"] for x in cell])

            results[fractional][logn] = {
                "total_err_err": total_err_err,
                "total_dist_err": total_dist_err}

    cell_width = 5
    header = ["", "Rate error", "Dist error"]
    header = ("{:{cell_width}s}" + 2 * " {:^{wide_width}s}").format(
        *header, cell_width=cell_width, wide_width=1 + 2 * cell_width)
    print(header)
    header = ["logn", "bin", "frac", "bin", "frac"]
    header = ("{:{cell_width}s}" + 4 * " {:{cell_width}s}").format(
        *header, cell_width=cell_width)
    print(header)
    for logn in logn_examples:
        frac_results = results[True][logn]
        bin_results = results[False][logn]
        row = [logn, bin_results["total_err_err"], frac_results["total_err_err"],
               bin_results["total_dist_err"], frac_results["total_dist_err"]]
        row = ("{:{cell_width}d}" + 4 * " {:{cell_width}.3f}").format(
            *row, cell_width=cell_width)
        print(row)


if __name__ == "__main__":
    main()
