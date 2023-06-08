import functions
import numpy as np
import scipy.optimize

runs = 30
dim = 20
bound = 100

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

#max_evals = 10_000 * dim  # like in the CEC'13 benchmark

cec_benchmark = functions.CEC_functions(dim)
bounds = dim * [(-bound, bound)]
popsize_DE = 50

#max_iter = int(max_evals / (popsize_DE * dim))

fun_num = 19


def obj_function(X):
    if len(X.shape) > 1:
        return cec_benchmark.Y_matrix(X, fun_num)
    else:
        return cec_benchmark.Y(X, fun_num)


# used for different max_evals
for i in range(1, 11):
    results = 0
    max_evals = 5000 * i * dim
    max_iter = int(max_evals / (popsize_DE * dim))
    for run in range(0, runs):
        #print(max_evals,max_iter)
        de = scipy.optimize.differential_evolution(obj_function, bounds=bounds, maxiter=max_iter, polish=False, tol=8)
        results += de.fun

    results = results / runs
    print(f"Function {fun_num}, DE result with max_evals = {max_evals} (error respect to the global optimum): {(results-fDeltas[fun_num-1]):.2E}")


# #used for different functions
# fun_num = 0
# results = np.zeros(29)
# for func_num in range(1, 29):
#     fun_num = func_num
#     for run in range(0, runs):
#         de = scipy.optimize.differential_evolution(obj_function, bounds=bounds, maxiter=max_iter, polish=False, tol=8)
#         results[func_num-1] += de.fun
#
#     results[func_num-1] = results[func_num-1] / runs
#     print(f"Function {func_num}, DE result (error respect to the global optimum): {(results[func_num-1]-fDeltas[func_num-1]):.2E}")
#
# np.save(f"results_de.np", results)