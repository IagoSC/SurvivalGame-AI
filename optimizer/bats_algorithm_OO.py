import numpy as np
from typing import Callable, Optional, Tuple

from dataclasses import dataclass

@dataclass
class BatsConfig:
    space_dimensions: int = 5
    population_size: int = 100
    max_iterations: int = 1000
    alpha: float = 0.9 # Step size for velocity update
    freq_interval: Tuple[float, float] = (0.1, 1.0)

    # TODO Use interval or uniform ?????
    pulse_rate_interval: Tuple[float, float] = (0.1, 1.0)
    initial_loudness_interval: Tuple[float, float] = (1.0, 2.0)
    initial_velocity_interval: Tuple[float, float] = (1.0, 2.0)

class Bat:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, frequency: float, loudness: float, pulse_rate: float):
        self.position = position
        self.velocity = velocity
        self.frequency = frequency
        self.loudness = loudness
        self.pulse_rate = pulse_rate

    def update_position(self):
        self.position += self.velocity

    def update_velocity(self, current_best_position: np.ndarray, alpha: float):
        self.velocity = (self.velocity + alpha * (current_best_position - self.position))

    def update_frequency(self, beta: float):
        self.frequency = np.random.uniform(self.frequency - 0.1, self.frequency + 0.1)
        self.frequency = np.clip(self.frequency, 0, 1)


class BatsAlgorithm:
    def __init__(self, config: BatsConfig, population_size: int = 100):
        self.config = config
        self.population_size = population_size
        self.population = self._initialize_population()
        self.current_best: Bat = None
        self.current_best_fitness = -float('inf')
        self.fitness_history = []

    def _initialize_population(self) -> list[Bat]:
        return [
            Bat(
                position=np.random.uniform(-1, 1, self.config.space_dimensions),
                velocity=np.random.uniform(-1, 1, self.config.space_dimensions),
                frequency=np.random.uniform(*self.config.freq_interval),
                loudness=np.random.uniform(*self.config.loud_interval),
                pulse_rate=np.random.uniform(*self.config.pulse_rate_interval)
            ) for _ in range(self.population_size)
        ]
    
    def initial_best_bat(self): 
        self.current_best = max(self.population, key=lambda bat: bat.loudness)
        self.current_best_fitness = self.current_best.loudness
    
    def find_best_bat(self):
        for bat in self.population:
            if bat.loudness > self.current_best_fitness:
                self.current_best = bat
                self.current_best_fitness = bat.loudness

    def optimize(self, fitness_function: Callable[[np.ndarray], float], bounds: Optional[tuple] = None):
        for iteration in range(self.config.max_iterations):
            for i, bat in enumerate(self.population):
                bat.update_frequency(self.betas[i])
                bat.update_velocity(self.current_best.position, self.config.alpha)
                bat.update_position()

                # Apply bounds
                if bounds is not None:
                    bat.position = np.clip(bat.position, bounds[0], bounds[1])

                # Evaluate fitness
                fitness = fitness_function(bat.position)
                if fitness > self.current_best_fitness:
                    self.current_best = bat
                    self.current_best_fitness = fitness

            self.fitness_history.append(self.current_best_fitness)

        return self.current_best.position