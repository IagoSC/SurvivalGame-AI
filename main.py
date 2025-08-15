import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent, neuralNetworkLayers
from optimizer.bats_algorithm import BatsAlgorithm
import os
import sys
import time
from test_trained_agent import test_agent
import db 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 100
GENERATIONS = 1_000
RENDER_TRAIN = False
RENDER_TEST = True
COLLECT_STATS = True
USE_BIAS = True 

game_config = GameConfig(num_players=POPULATION_SIZE)


def game_fitness_function(population: np.ndarray) -> float:
    agents = [
        NeuralNetworkAgent(
            grid_size=game_config.sensor_grid_size,
            network_setup=weights,
            useBias=USE_BIAS
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
    
    db.init_db()
    
    _, n_dimensions = neuralNetworkLayers(grid_size=game_config.sensor_grid_size, useBias=USE_BIAS)

    ba = BatsAlgorithm(
        dimensions=n_dimensions,
        population_size=POPULATION_SIZE,
        max_gen=GENERATIONS,
    )

    best_weights_overall = None
    best_fitness_overall = -np.inf

    start_time = time.time()

    run_id = f"{start_time}_run"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    table_runs = "runs"
    table_bats = "bats"

    if COLLECT_STATS:
        db.add_to_table(table_runs, {
            "run_id": run_id,
            "timestamp": timestamp,
            **ba.get_params(),
        })

    for current_gen in range(GENERATIONS):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 60 * 60 * 12:  # 12 hours
            break

        if COLLECT_STATS:
            last_positions = ba.positions

        best_solution, best_fitness = ba.move_bats(
            fitness_function=game_fitness_function,
            gen_idx=current_gen
        )

        if COLLECT_STATS:
            loudness, positions, fitness = ba.loudness, ba.positions, ba.current_gen_fitness

            dist_between_bats = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)

            # Movement can be zero if no bats moved, but new positions were not accepted
            movement = np.linalg.norm(positions - last_positions, axis=1)
            
            db.add_to_table(table_bats, {
                "run_id": run_id,
                "generation": current_gen,
                "best_fitness": best_fitness,
                # "best_solution": f"{best_solution.tolist()}",
                # "fitness": f"{fitness.tolist()}",
                "avg_fitness": np.mean(fitness),
                "std_fitness": np.std(fitness),
                "avg_loudness": np.mean(loudness),
                "std_loudness": np.std(loudness),
                "avg_pulse_rate": np.mean(ba.pulse_rate),
                "std_pulse_rate": np.std(ba.pulse_rate),
                "avg_distance_between_bats": np.mean(dist_between_bats),
                "std_distance_between_bats": np.std(dist_between_bats),
                "avg_movement": np.mean(movement),
                "std_movement": np.std(movement),
                "avg_frequency": np.mean(ba.frequencies),
                "std_frequency": np.std(ba.frequencies),
            })

    db.close_connection()
    best_fitness_overall = best_fitness
    best_weights_overall = best_solution

    print(f"Melhor Fitness Inicial: {best_fitness_overall:.2f}")

    print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall:.2f}')
    np.save("best_weights.npy", best_weights_overall)
    
    print(f"Melhor Fitness Geral: {best_fitness_overall:.2f}")
   

    print("\n--- Treinamento Concluído ---")

    if best_weights_overall is not None:
        np.save(f"best_weights_{run_id}.npy", best_weights_overall)
        print("Melhores pesos salvos em \'best_weights.npy\'")
 
        test_agent(best_weights_overall, num_tests=30, render=RENDER_TEST)
    else:
        print("Nenhum peso ótimo encontrado.")

if __name__ == "__main__":
    # best_weights_overall = np.load("best_weights.npy")
    # test_agent(best_weights_overall, num_tests=30, render=False)
    train_and_test()