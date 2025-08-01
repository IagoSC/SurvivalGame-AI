import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

# Set backend for non-interactive environments
try:
    matplotlib.use('TkAgg')  # Try interactive backend first
except:
    matplotlib.use('Agg')    # Fall back to non-interactive backend

np.random.seed(100)


class BatsAlgorithm:
    def __init__(self, dimensions = 8, population_size = 100, max_gen = 1000, lower_bound = -1.0, upper_bound = 1.0, min_freq = 0.0, max_freq = 1.0, min_pulse = 0.005, max_pulse = 0.5, min_A = 0.5, max_A = 1.5, alpha = 0.9, gamma = 0.9, plot_evolution = False):
        self.population_size = population_size
        self.dimensions = dimensions
        self.max_gen = max_gen
        self.alpha = alpha
        self.gamma = gamma
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.plot_evolution = plot_evolution

        self.positions = np.random.uniform(lower_bound, upper_bound,  (population_size, dimensions))
        self.velocities = np.zeros((population_size, dimensions)) 
        self.frequencies = np.random.uniform(min_freq, max_freq, population_size)
        self.loudness = np.random.uniform(min_A, max_A, population_size)
        self.initial_pulse_rate = np.random.uniform(min_pulse, max_pulse, population_size)
        self.pulse_rate = self.initial_pulse_rate

        if self.plot_evolution:
            self.position_history = []
            self.fitness_history = []
            self.best_fitness_history = []

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
        self.pulse_rate = self.initial_pulse_rate * (1 - math.exp(-self.gamma * current_iteration))

    def _fly_randomly(self, best_position: np.ndarray, current_positions: np.ndarray, avg_loudness: float) -> np.ndarray:
        rand = np.random.uniform(0,1, self.population_size)
        is_moving = np.greater(rand,  self.pulse_rate)

        best_positions = np.array([best_position] * self.population_size)

        place_of_search = np.where(
            is_moving[:, np.newaxis],
            best_positions,
            self.positions + self.velocities
        )

        epsilon = np.random.uniform(-1, 1, (self.population_size, self.dimensions)) * avg_loudness
        return np.clip(place_of_search +  epsilon, self.lower_bound, self.upper_bound)

    def optimize(self, fitness_function) -> np.ndarray:
        self._evaluate_population_fitness(fitness_function)
        
        for gen in range(self.max_gen):
            if self.plot_evolution:
                self.position_history.append(self.positions.copy())
                self.fitness_history.append(self.current_gen_fitness.copy())
                self.best_fitness_history.append(self.current_gen_fitness[self.current_best_idx])

            best_position = self.positions[self.current_best_idx]
            self._update_frequencies()
            self._update_velocities(best_position)
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

        # Plot evolution if requested - now saves files instead of showing
        if self.plot_evolution:
            self.plot_position_evolution()
            self.plot_fitness_evolution()

        return self.positions[self.current_best_idx], self.current_gen_fitness[self.current_best_idx]

    def plot_position_evolution(self, save_path=None):
        """Plot the evolution of bat positions over generations."""
        if not hasattr(self, 'position_history'):
            print("No position history available. Set plot_evolution=True during initialization.")
            return
            
        # Plot first two dimensions for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot position scatter for first and last generation
        gen_first = 0
        gen_last = len(self.position_history) - 1
        
        # First generation
        axes[0, 0].scatter(self.position_history[gen_first][:, 0], 
                          self.position_history[gen_first][:, 1], 
                          c=self.fitness_history[gen_first], 
                          cmap='viridis', alpha=0.6)
        axes[0, 0].set_title(f'Generation {gen_first + 1} - Positions')
        axes[0, 0].set_xlabel('Dimension 1')
        axes[0, 0].set_ylabel('Dimension 2')
        axes[0, 0].grid(True)
        
        # Last generation
        scatter = axes[0, 1].scatter(self.position_history[gen_last][:, 0], 
                                   self.position_history[gen_last][:, 1], 
                                   c=self.fitness_history[gen_last], 
                                   cmap='viridis', alpha=0.6)
        axes[0, 1].set_title(f'Generation {gen_last + 1} - Positions')
        axes[0, 1].set_xlabel('Dimension 1')
        axes[0, 1].set_ylabel('Dimension 2')
        axes[0, 1].grid(True)
        plt.colorbar(scatter, ax=axes[0, 1], label='Fitness')
        
        # Best bat trajectory over time
        best_positions = []
        for gen_positions, gen_fitness in zip(self.position_history, self.fitness_history):
            best_idx = np.argmax(gen_fitness)
            best_positions.append(gen_positions[best_idx])
        best_positions = np.array(best_positions)
        
        axes[1, 0].plot(best_positions[:, 0], best_positions[:, 1], 'r-', alpha=0.7, linewidth=2)
        axes[1, 0].scatter(best_positions[0, 0], best_positions[0, 1], c='green', s=100, label='Start', marker='o')
        axes[1, 0].scatter(best_positions[-1, 0], best_positions[-1, 1], c='red', s=100, label='End', marker='s')
        axes[1, 0].set_title('Best Bat Trajectory')
        axes[1, 0].set_xlabel('Dimension 1')
        axes[1, 0].set_ylabel('Dimension 2')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Population spread over time
        spreads = []
        for positions in self.position_history:
            spread = np.std(positions, axis=0)
            spreads.append(np.mean(spread))
        
        axes[1, 1].plot(range(len(spreads)), spreads, 'b-', linewidth=2)
        axes[1, 1].set_title('Population Spread Over Time')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Average Standard Deviation')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save or show based on environment
        if save_path or matplotlib.get_backend() == 'Agg':
            save_file = save_path or 'position_evolution.png'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Position evolution plot saved to {save_file}")
        else:
            plt.show()
        
        plt.close()  # Free memory

    def plot_fitness_evolution(self, save_path=None):
        """Plot fitness evolution over generations."""
        if not hasattr(self, 'fitness_history'):
            print("No fitness history available. Set plot_evolution=True during initialization.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        generations = range(len(self.best_fitness_history))
        avg_fitness = [np.mean(fitness) for fitness in self.fitness_history]
        std_fitness = [np.std(fitness) for fitness in self.fitness_history]
        
        # Best fitness over time
        axes[0, 0].plot(generations, self.best_fitness_history, 'r-', linewidth=2, label='Best')
        axes[0, 0].plot(generations, avg_fitness, 'b-', linewidth=2, label='Average')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Fitness standard deviation
        axes[0, 1].plot(generations, std_fitness, 'g-', linewidth=2)
        axes[0, 1].set_title('Fitness Standard Deviation')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Std Deviation')
        axes[0, 1].grid(True)
        
        # Fitness distribution for first and last generation
        axes[1, 0].hist(self.fitness_history[0], bins=20, alpha=0.7, label=f'Gen 1')
        axes[1, 0].hist(self.fitness_history[-1], bins=20, alpha=0.7, label=f'Gen {len(self.fitness_history)}')
        axes[1, 0].set_title('Fitness Distribution')
        axes[1, 0].set_xlabel('Fitness')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Convergence rate (improvement per generation)
        improvements = np.diff(self.best_fitness_history)
        axes[1, 1].plot(range(1, len(improvements) + 1), improvements, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Fitness Improvement per Generation')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Fitness Improvement')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save or show based on environment
        if save_path or matplotlib.get_backend() == 'Agg':
            save_file = save_path or 'fitness_evolution.png'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Fitness evolution plot saved to {save_file}")
        else:
            plt.show()
        
        plt.close()  # Free memory

    def create_animation(self, interval=200, save_path='bat_animation.gif'):
        """Create an animated plot showing position evolution."""
        if not hasattr(self, 'position_history'):
            print("No position history available. Set plot_evolution=True during initialization.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            positions = self.position_history[frame]
            fitness = self.fitness_history[frame]
            
            scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                               c=fitness, cmap='viridis', alpha=0.6, s=50)
            
            # Highlight best bat
            best_idx = np.argmax(fitness)
            ax.scatter(positions[best_idx, 0], positions[best_idx, 1], 
                      c='red', s=200, marker='*', edgecolors='black')
            
            ax.set_xlim(self.lower_bound, self.upper_bound)
            ax.set_ylim(self.lower_bound, self.upper_bound)
            ax.set_title(f'Generation {frame + 1} - Best Fitness: {fitness[best_idx]:.4f}')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True)
            
            return scatter,
        
        anim = FuncAnimation(fig, animate, frames=len(self.position_history), 
                           interval=interval, blit=False, repeat=True)
        
        # Always save animation
        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")
        
        plt.close()  # Free memory
        return anim