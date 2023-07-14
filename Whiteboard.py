# Testing function ideas

import numpy as np
import itertools
import time

def generate_error_rates(mean, variance, combinations):
    combination_list = itertools.product(*[range(2) for _ in range(full_dim)])
    error_rates = np.zeros(combinations.shape[0])
    
    while True:
        # Generate random error rates
        error_rates = np.random.normal(mean, np.sqrt(variance), combinations.shape[0])
        
        # Calculate the mean and variance of the generated error rates
        generated_mean = np.mean(error_rates)
        generated_variance = np.var(error_rates)
        
        # Check if the generated mean and variance match the desired values within a tolerance
        if np.abs(generated_mean - mean) < 0.001 and np.abs(generated_variance - variance) < 0.001:
            break
    
    # Assign the generated error rates to each combination
    combinations_with_error_rates = np.column_stack((combinations, error_rates))
    
    return combinations_with_error_rates

def normal_error_rates(full_dim, classifier_error):
    true_errs = {}
    combos = list(itertools.product(*[range(2) for _ in range(full_dim)]))
    diff_level = 10
    indices = [i for i in range(diff_level)]
    for assn in itertools.product(*[range(2) for _ in range(len(indices))]):
      err = np.random.normal(classifier_error, classifier_error / 4)
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


    print(true_errs)

start_time = time.time()
normal_error_rates(15, .5)
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")