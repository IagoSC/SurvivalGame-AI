import math
import numpy as np

np.random.seed(100)

class BatsAlgorithm:
    def __init__(self, dimensions = 8, population_size = 100, max_gen = 1000, lower_bound = -1.0, upper_bound = 1.0, min_freq = 0.0, max_freq = 1.0, min_pulse = 0.005, max_pulse = 0.5, min_A = 0.5, max_A = 1.5, alpha = 0.9, gamma = 0.9):
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_gen = max_gen
        self.alpha = alpha
        self.gamma = gamma
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.positions = np.random.uniform(lower_bound, upper_bound,  (population_size, dimensions))
        self.velocities = np.zeros((population_size, dimensions)) 
        self.frequencies = np.random.uniform(min_freq, max_freq, population_size)
        self.loudness = np.random.uniform(min_A, max_A, population_size)
        self.initial_pulse_rate = np.random.uniform(min_pulse, max_pulse, population_size)
        self.pulse_rate = self.initial_pulse_rate

    def _evaluate_population_fitness(self, fitness_function) -> np.ndarray:
        self.current_gen_fitness = fitness_function(self.positions)
        self.current_best_idx = np.argmax(self.current_gen_fitness)
        
    def _update_frequencies(self):
        beta = np.random.uniform(0, 1, self.population_size)
        self.frequencies = self.min_freq + (self.max_freq - self.min_freq) * beta

    def _update_velocities(self, current_best_position: np.ndarray):
        direction = (current_best_position - self.positions)
        self.velocities += direction * self.frequencies[:, np.newaxis]

    def _update_positions(self):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

    def _update_loudness(self):
        self.loudness *= self.alpha

    def _update_pulse_rate(self, current_iteration: int):
        self.pulse_rate = self.initial_pulse_rate * (1 - (math.exp(-self.gamma * current_iteration)))

    def _fly_randomly(self, best_position: np.ndarray, current_positions: np.ndarray, avg_loudness: float) -> np.ndarray:
        rand = np.random.uniform(0,1, self.population_size)
        is_moving = np.greater(rand,  self.pulse_rate)

        epsilon = np.where(
            is_moving[:, np.newaxis],
            np.random.uniform(-1, 1, current_positions.shape) * avg_loudness,
            np.zeros(current_positions.shape)
        )
        best_positions = np.array([best_position] * self.population_size)
        return np.clip(best_positions +  epsilon, self.lower_bound, self.upper_bound)

    def optimize(self, fitness_function) -> np.ndarray:
        self._evaluate_population_fitness(fitness_function)
        
        for gen in range(self.max_gen):
            best_position = self.positions[self.current_best_idx]
            self._update_frequencies()
            self._update_velocities(best_position)
            self._update_positions()
            self._evaluate_population_fitness(fitness_function)

            # TODO 
            # 1 - Try move anyway
            # 2 - Try move only if fitness is better and loudnes check

            avg_loudness = np.mean(self.loudness)

            new_positions = self._fly_randomly(best_position, self.positions, avg_loudness)

            new_fitness = fitness_function(new_positions)

            is_better = np.greater_equal(new_fitness, self.current_gen_fitness)
            is_below_loudness = np.less(np.random.uniform(0, 1, self.population_size), self.loudness)

            update_positions = np.logical_and(is_better, is_below_loudness)

            self.positions = np.where(
                update_positions[:, np.newaxis],
                new_positions,
                self.positions
            )

            self.current_gen_fitness = np.where(
                update_positions,
                new_fitness,
                self.current_gen_fitness
            )

            self._update_loudness()
            self._update_pulse_rate(gen)
            self.current_best_idx = np.argmax(self.current_gen_fitness)
            print(f"Generation {gen + 1}")
            print(f"Best fitness {self.current_gen_fitness[self.current_best_idx]}")
            print(f"Avg fitness {np.mean(self.current_gen_fitness)}")
            print(f"Std fitness {np.std(self.current_gen_fitness)}")

        return self.positions[self.current_best_idx], self.current_gen_fitness[self.current_best_idx]