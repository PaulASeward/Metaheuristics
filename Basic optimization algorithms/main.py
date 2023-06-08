import scipy.optimize
import functions
from minimum_population_search import *
import numpy as np

# These are the fitness values of the global optima for the 28 functions in the benchmark.
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

dim = 20
max_evals = 10_000 * dim  # like in the CEC'13 benchmark
cec_benchmark = functions.CEC_functions(dim)

bound = 100
bounds = dim * [(-bound, bound)]


def obj_function(X):
    if len(X.shape) > 1:
        return cec_benchmark.Y_matrix(X, fun_num)
    else:
        return cec_benchmark.Y(X, fun_num)


fun_num = 9

# popsize_DE = 50
# max_iter = int(max_evals / (popsize_DE * dim))
#
# results = scipy.optimize.differential_evolution(obj_function, bounds=bounds, maxiter=max_iter, polish=False, tol=8)
# print(f"Function {fun_num}, DE result (error from optimum): {(results.fun - fDeltas[fun_num - 1]):.2E}")



best_solution, best_fitness = mps(obj_function, dim=dim, max_eval=max_evals,
                                  pop_size=100, bound=bound)
print(f"Function {fun_num}, MPS result (error from optimum): {(best_fitness - fDeltas[fun_num - 1]):.2E}")