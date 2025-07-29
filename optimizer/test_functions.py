import numpy as np
from bats_algorithm import BatsAlgorithm as BA1, BatsConfig
from another_bats_algorithm import BatsAlgorithm as BA2
def sphere_function(x):
    """
    Sphere function.
    Global minimum at x = [0, 0, 0, 0, 0], f(x) = 0
    """
    return np.sum(np.square(x))

def rastrigin_function(x):
    """
    Rastrigin function.
    Global minimum at x = [0, 0, 0, 0, 0], f(x) = 0
    """
    A = 10
    return A * len(x) + np.sum(np.square(x) - A * np.cos(2 * np.pi * x))

def ackley_function(x):
    """
    Ackley function.
    Global minimum at x = [0, 0, 0, 0, 0], f(x) = 0
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum_sq = np.sum(np.square(x))
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)



if __name__ == "__main__":
    # Example usage
    ba = BA1(
        BatsConfig(
            space_dimensions=5,
            space_lower_bound=-5.0,
            space_upper_bound=5.0,
            population_size=10,
            max_iterations=1000,
            alpha=0.9,
            gamma=0.9,
            min_freq=0.0,
            max_freq=1.0,
            min_A=0.0,
            max_A=1.0,
            min_pulse_rate=0.0,
            max_pulse_rate=1.0
        )
    )

    ba2 = BA2(
        num_bats = 30,
        dim = 10,
        num_iterations = 100,
        freq_min = 0,
        freq_max = 2,
        A = 0.5,
        r0 = 0.5,
        alpha = 0.9,
        gamma = 0.9,
        lb = -10,
        ub = 10,
    )

    ba2.optimize(sphere_function)
    print("Best position:", ba.positions[ba.current_best_idx])
    print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

    # ba.optimize(rastrigin_function)
    # print("Best position:", ba.positions[ba.current_best_idx])
    # print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

    # ba.optimize(ackley_function)
    # print("Best position:", ba.positions[ba.current_best_idx])
    # print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

