import numpy as np
from bats_algorithm_lib import BatAlgorithm as BA

def sphere_function(d, x):
    """
    Sphere function.
    Global minimum at x = [0, 0, 0, 0, 0], f(x) = 0
    """
    solutions = [np.sum(np.square(el)) for el in x]
    return solutions

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
    ba = BA(250, 40, 100, 1, 0.2, 0.0, 3.0, -10, 10, sphere_function)
    ba.move_bat()

    print("Best position:", ba.best)
    print("Best fitness:", ba.f_min)


    # ba2.optimize(sphere_function)
    # print("Best position:", ba.positions[ba.current_best_idx])
    # print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

    # ba.optimize(rastrigin_function)
    # print("Best position:", ba.positions[ba.current_best_idx])
    # print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

    # ba.optimize(ackley_function)
    # print("Best position:", ba.positions[ba.current_best_idx])
    # print("Best fitness:", ba.current_gen_fitness[ba.current_best_idx])

