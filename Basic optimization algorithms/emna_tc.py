import numpy as np


def emna_tc(fun, dim, max_eval, pop_size, bound):

    # EMNA parameters
    sel_coef = 0.2
    gamma = 3

    # Initial population
    population = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    f_pop = fun(population)
    count_eval = pop_size

    best_idx = np.argmin(f_pop)
    best_f = f_pop[best_idx]
    best = population[best_idx, :]

    init_threshold = None

    while count_eval < max_eval:
        arg_sorted = np.argsort(f_pop)
        ref_sols = population[arg_sorted[:int(pop_size*sel_coef)]]

        if init_threshold is None:
            init_threshold = np.linalg.norm(np.cov(ref_sols, rowvar=False))
        threshold = np.maximum(init_threshold * (np.power((max_eval - count_eval) / max_eval, gamma)), 1e-05)

        sigma = np.cov(ref_sols, rowvar=False)
        sigma = threshold * (sigma/np.linalg.norm(sigma))  # Applying TC
        miu = np.mean(ref_sols, axis=0)

        population = np.random.multivariate_normal(miu, sigma, pop_size)
        f_pop = fun(population)
        count_eval += pop_size

        best_idx = np.argmin(f_pop)
        if best_f > f_pop[best_idx]:
            best_f = f_pop[best_idx]
            best = population[best_idx, :]

    return best, best_f

