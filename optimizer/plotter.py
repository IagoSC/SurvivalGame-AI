import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np

# Set backend for non-interactive environments
try:
    matplotlib.use('TkAgg')  # Try interactive backend first
except:
    matplotlib.use('Agg')    # Fall back to non-interactive backend

class Plotter:
    def __init__(self, lower_bound=-10, upper_bound=10):
        self.position_history = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def add_history(self, positions, fitness, best_fitness):
        self.position_history.append(positions.copy())
        self.fitness_history.append(fitness.copy())
        self.best_fitness_history.append(best_fitness)


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