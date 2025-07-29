import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent, neuralNetworkLayers
from optimizer.bats_algorithm_lib import BatAlgorithm
import os
import sys
import time

from optimizer.bats_algorithm import BatsConfig
from test_trained_agent import test_agent

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 500
GENERATIONS = 100
RENDER_TRAIN = False
RENDER_TEST = True


game_config = GameConfig(num_players=POPULATION_SIZE,fps=120)


def game_fitness_function(population: np.ndarray) -> float:
    agents = [
        NeuralNetworkAgent(
            grid_size=game_config.sensor_grid_size,
            network_setup=weights
        ) for weights in population
    ] 

    total_scores = np.zeros(len(agents))
    for i in range(3):
        game = SurvivalGame(config=game_config, render=RENDER_TRAIN)
        while not game.all_players_dead():
            actions = []
            for idx, agent in enumerate(agents):
                if game.players[idx].alive:
                    state = game.get_state(idx, include_internals=True)
                    action = agent.predict(state)
                    actions.append(action)
                else:
                    actions.append(0)

            game.update(actions)
            if game.render:
                game.render_frame()
        for idx, player in enumerate(game.players):
            total_scores[idx] += player.score

    average_scores = total_scores / 3
    print(f"Melhor: {np.max(average_scores):.2f} | Média: {np.mean(average_scores):.2f} | Std: {np.std(average_scores):.2f}")
    return average_scores

def train_and_test():
    print("\n--- Iniciando Treinamento com Algoritmo Genético ---")

    n_dimensions = sum(layer_in * layer_out + layer_out for layer_in, layer_out in neuralNetworkLayers(grid_size=game_config.sensor_grid_size))
    
    ba_config = BatsConfig(
        space_dimensions=n_dimensions,
        space_lower_bound=-1.0,
        space_upper_bound=1.0,
        population_size=POPULATION_SIZE,
        max_iterations=GENERATIONS,
        alpha=0.9,
        gamma=0.9,
        min_freq=0.0,
        max_freq=0.8,
        min_A=0.0,
        max_A=1.0,
        min_pulse_rate=0.0,
        max_pulse_rate=1.0
    )

    ba = BatAlgorithm(
        ba_config.space_dimensions,
        ba_config.population_size,
        ba_config.max_iterations,
        0.9,
        0.2,
        ba_config.min_freq,
        ba_config.max_freq,
        ba_config.space_lower_bound,
        ba_config.space_upper_bound,
        lambda x, y: game_fitness_function(y),
    )

    best_weights_overall = None
    best_fitness_overall = -np.inf

    ba.move_bat()

    best_fitness_overall = ba.f_max
    best_weights_overall = ba.best
    print(f"Melhor Fitness Inicial: {best_fitness_overall:.2f}")

    print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall:.2f}')
    np.save("best_weights.npy", best_weights_overall)
    
    print(f"Melhor Fitness Geral: {best_fitness_overall:.2f}")
   

    print("\n--- Treinamento Concluído ---")

    if best_weights_overall is not None:
        np.save("best_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em \'best_weights.npy\'")
 
        test_agent(best_weights_overall, num_tests=30, render=RENDER_TEST)
    else:
        print("Nenhum peso ótimo encontrado.")

if __name__ == "__main__":
    train_and_test()