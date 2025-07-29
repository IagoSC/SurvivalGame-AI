import math
import numpy as np
from typing import Callable, Optional, Tuple

from dataclasses import dataclass

@dataclass
class BatsConfig:
    space_dimensions: int = 1
    space_lower_bound: float = -1.0
    space_upper_bound: float = 1.0
    population_size: int = 10
    max_iterations: int = 1000
    alpha: float = 0.9 #
    gamma: float = 0.9 # Loudness decay factor
    min_freq: float = 0.0
    max_freq: float = 1.0
    min_A: float = 0.0
    max_A: float = 1.0
    min_pulse_rate: float = 0.0
    max_pulse_rate: float = 1.0

class BatsAlgorithm:
    def __init__(self, config: BatsConfig):
        self.config = config
        self.population_size = config.population_size
        self.positions = np.random.uniform(self.config.space_lower_bound, self.config.space_upper_bound,  (self.population_size, self.config.space_dimensions))
        # Since we dont have initial best, there's no need for initial velocities
        self.velocities = np.zeros((self.population_size, self.config.space_dimensions)) 
        self.frequencies = np.random.uniform(config.min_freq, config.max_freq, self.population_size)
        self.loudness = np.random.uniform(config.min_A, (config.min_A + config.max_A)/2, self.population_size)
        self.initial_pulse_rate = np.random.uniform(config.min_pulse_rate, (config.min_pulse_rate + config.max_pulse_rate)/2, self.population_size)
        self.pulse_rate = self.initial_pulse_rate
        self.current_best_idx: int = -1
        self.current_gen_fitness: np.ndarray = float('inf') * np.ones(self.population_size)

    def _evaluate_population_fitness(self, fitness_function: Callable[[np.ndarray], float]) -> np.ndarray:
        self.current_gen_fitness = [fitness_function(position) for i, position in enumerate(self.positions)]
        self.current_best_idx = np.argmin(self.current_gen_fitness).__int__()
        
    def _update_frequencies(self):
        beta = np.random.uniform(0, 1, self.population_size)
        self.frequencies = np.clip(self.config.min_freq + (self.config.max_freq - self.config.min_freq) * beta)

    def _update_velocities(self, current_best_position: np.ndarray, alpha: float):
        direction = (current_best_position - self.positions)
        self.velocities += direction * self.frequencies[:, np.newaxis]

    def _update_positions(self):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.config.space_lower_bound, self.config.space_upper_bound)

    def _update_loudness(self):
        self.loudness *= self.config.alpha

    def _update_pulse_rate(self, current_iteration: int):
        self.pulse_rate = self.pulse_rate * (1 - (math.exp(-self.config.gamma * current_iteration)))

    def _fly_randomly(self, position: np.ndarray, epsilon: float, avg_loudness: float) -> np.ndarray:
        return np.clip(position + epsilon * avg_loudness, self.config.space_lower_bound, self.config.space_upper_bound)

    def optimize(self, fitness_function: Callable[[np.ndarray], float]) -> np.ndarray:
        self._evaluate_population_fitness(fitness_function)
        
        for gen in range(self.config.max_iterations):
            best_position = self.positions[self.current_best_idx]
            self._update_frequencies()
            self._update_velocities(best_position, self.config.alpha)
            self._update_positions()
            self._evaluate_population_fitness(fitness_function)
            
            avg_loudness = np.mean(self.loudness)
            
            for i in range(self.population_size):
                epsilon = np.random.uniform(-1, 1, self.config.space_dimensions)

                if np.random.uniform(0,1) > self.pulse_rate[i]:
                    # Search around global best
                    new_position = self._fly_randomly(best_position, epsilon, avg_loudness)
                else:
                    # Local search
                    new_position = self._fly_randomly(self.positions[i], epsilon, avg_loudness)

                new_fitness = fitness_function(new_position)
                if new_fitness <= self.current_gen_fitness[self.current_best_idx] and np.random.uniform(0, 1) < self.loudness[i]:
                    self.positions[i] = new_position
                    self.current_gen_fitness[i] = new_fitness
                    
                    if new_fitness < self.current_gen_fitness[self.current_best_idx]:
                        self.current_best_idx = i
                
                self.loudness[i] *= self.config.alpha
                self.pulse_rate[i] = np.random.uniform(self.config.min_pulse_rate, self.config.max_pulse_rate)


        return self.positions[self.current_best_idx]