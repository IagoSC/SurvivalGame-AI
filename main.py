import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent, neuralNetworkLayers
from optimizer.bats_algorithm import BatsAlgorithm
import os
import sys
import time
from test_trained_agent import test_agent

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 500
GENERATIONS = 10_000
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

    n_runs = 3
    best_run = 0
    for i in range(n_runs):
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
            if best_run < player.score:
                best_run = player.score
            total_scores[idx] += player.score

    average_scores = total_scores / n_runs

    print(f"Melhor Run: {best_run:.2f} | Melhor Individuo: {np.max(average_scores):.2f} | Média: {np.mean(average_scores):.2f} | Std: {np.std(average_scores):.2f}")
    return average_scores

def train_and_test():
    print("\n--- Iniciando Treinamento com Algoritmo Genético ---")

    n_dimensions = sum(layer_in * layer_out + layer_out for layer_in, layer_out in neuralNetworkLayers(grid_size=game_config.sensor_grid_size))

    ba = BatsAlgorithm(
        dimensions=n_dimensions,
        population_size=POPULATION_SIZE,
        max_gen=GENERATIONS,
    )

    best_weights_overall = None
    best_fitness_overall = -np.inf

    ba.optimize(
        fitness_function=game_fitness_function,
    )

    best_fitness_overall = ba.best_fitness
    best_weights_overall = ba.best_solution

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