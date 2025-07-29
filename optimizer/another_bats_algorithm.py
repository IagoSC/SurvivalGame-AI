import numpy as np


class BatsAlgorithm:
    def __init__(self, num_bats, dim, num_iterations, freq_min, freq_max, A, r0, alpha, gamma, lb, ub):
        self.num_bats = num_bats
        self.dim = dim
        self.num_iterations = num_iterations
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.A = A
        self.r0 = r0
        self.alpha = alpha
        self.gamma = gamma
        self.lb = lb
        self.ub = ub

        # Initialize bat positions and velocities
        self.positions = np.random.uniform(lb, ub, (num_bats, dim))
        self.velocities = np.zeros((num_bats, dim))
        self.frequencies = np.zeros(num_bats)
        self.loudness = A * np.ones(num_bats)
        self.pulse_rate = r0 * np.ones(num_bats)
        self.fitness = np.full(num_bats, np.inf)

    def optimize(self, function):
        fitness = np.apply_along_axis(function, 1, self.positions)
        self.best_position = self.positions[np.argmin(fitness)]
        self.best_fitness = np.min(fitness)

        for iteration in range(self.num_iterations):
            avg_loudness = np.mean(self.loudness)
            avg_pulse_rate = np.mean(self.pulse_rate)

            # Update bats
            for i in range(self.num_bats):
                beta = np.random.uniform(0, 1)
                self.frequencies[i] = self.freq_min + (self.freq_max - self.freq_min) * beta
                self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
                new_position = self.positions[i] + self.velocities[i]

                # Boundary check
                new_position = np.clip(new_position, self.lb, self.ub)

                # Local search
                if np.random.uniform(0, 1) > self.pulse_rate[i]:
                    epsilon = np.random.uniform(-1, 1)
                    new_position = self.positions[i] + epsilon * avg_loudness
                
                # Evaluate new solution
                new_fitness = function(new_position)
                
                # Greedy mechanism to update if new solution is better and random value is less than loudness
                if new_fitness < self.fitness[i] and np.random.uniform(0, 1) < self.loudness[i]:
                    self.positions[i] = new_position
                    self.fitness[i] = new_fitness

                # Update global best
                if self.fitness[i] < self.best_fitness:
                    self.best_position = self.positions[i]
                    self.best_fitness = self.fitness[i]

                self.loudness[i] *= self.alpha
                self.pulse_rate[i] = self.r0 * (1 - np.exp(-self.gamma * iteration))


        print("\nOptimized Solution:", self.best_position)
        print("Best Fitness Value:", self.best_fitness)