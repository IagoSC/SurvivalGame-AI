import numpy as np
# from bats_algorithm import BatsConfig
from bats_algorithm import BatsAlgorithm as BA
from bats_algorithm_indonesio import BatAlgorithm as BAI

from plotter import Plotter

def maximization_benchmark_function(x):
    """
    Maximization benchmark function.
    The maximum value is at the origin (0, 0, ..., 0).
    Example: Negative sphere function shifted to be maximized at zero.
    """

    return [-np.sum(np.square(el)) for el in x]

def egg_crate_function(x):
    """
    Egg crate function for minimization.
    The minimum value is at the origin (0, 0, ..., 0).
    """
    return [-(np.sum([el**2 for el in xi]) + 25 * np.sum(np.sin(xi))) for xi in x]

if __name__ == "__main__":
    # Example usage
    ba = BA(
        population_size = 15,
        dimensions=2,
        max_gen=200,
        plot_evolution=True,
        lower_bound=-10,
        upper_bound=10,
        max_freq=1.0,
        plotter=Plotter
    )
    best_bat, best_fitness = ba.optimize(fitness_function=egg_crate_function )
    
    print("Best bat position:", best_bat)
    print("Best bat fitness:", best_fitness)

    ba.plot(interval=200, save_path="bat_evolution.gif")