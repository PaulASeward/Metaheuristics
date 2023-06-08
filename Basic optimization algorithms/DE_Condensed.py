import numpy as np
from random import random, randint

class DE:
    def __init__(self, func, bounds, pop_size, dimensions, max_iter, strategy='rand/1/bin', F=0.5, CR=0.9):  # Also try F=0.9
        self.func = func
        self.bounds = bounds
        self.pop_size = pop_size
        self.dimension = dimensions
        self.max_iter = max_iter
        self.strategy = strategy
        self.F = F  # Learning Factor
        self.CR = CR  # Crossover Constant

    def optimize(self):
        population = self._init_population()
        fitness = self._evaluate_population(population)

        for i in range(self.max_iter):
            trial_pop = self._generate_trial_population(population)
            trial_fitness = self._evaluate_population(trial_pop)

            population, fitness = self._select_population(population, fitness, trial_pop, trial_fitness)

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        return best_solution, best_fitness

    def _init_population(self):
        lower_bound, upper_bound = self.bounds
        population = np.empty((self.pop_size, self.dimension))

        for i in range(self.pop_size):
            for j in range(self.dimension):
                population[i, j] = lower_bound[j] + random() * (upper_bound[j] - lower_bound[j])

        return population
    
    
    
    
    # From Scipy.optimize
    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (M, len(x)), where M > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0
        
        
        
        
    

    def _evaluate_population(self, pop):
        return np.array([self.func(individual) for individual in pop])

    def _generate_trial_population(self, pop):
        trial_pop = np.empty_like(pop)

        for i in range(self.pop_size):
            a, b, c = self._select_parents(i, pop)
            trial_pop[i] = self._mutate(a, b, c)

        return trial_pop

    def _select_parents(self, current_idx, pop):
        if self.strategy == 'rand/1/bin':
            while True:
                idxs = [idx for idx in range(self.pop_size) if idx != current_idx]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                if a is not b is not c:
                    return a, b, c
        else:
            raise NotImplementedError(f"Strategy '{self.strategy}' is not implemented.")

    def _mutate(self, a, b, c):
        mutated = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
        return mutated

    def _select_population(self, pop, fitness, trial_pop, trial_fitness):
        selected = np.where(trial_fitness < fitness)
        pop[selected] = trial_pop[selected]
        fitness[selected] = trial_fitness[selected]
        return pop, fitness


def sphere(x):
    return np.sum(x**2)

bounds = (-100, 100)
pop_size = 50
max_iter = 1000

de = DE(func=sphere, bounds=bounds, pop_size=pop_size, max_iter=max_iter)
best_solution, best_fitness = de.optimize()
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)