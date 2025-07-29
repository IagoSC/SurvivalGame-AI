import numpy as np

# Objective function: Sphere function
def sphere_function(x):
    return np.sum(x**2)

# Parameters
num_bats = 30
dim = 10
num_iterations = 100
freq_min = 0
freq_max = 2
A = 0.5
r0 = 0.5
alpha = 0.9
gamma = 0.9
lb = -10
ub = 10

# Initialize bat positions and velocities
positions = np.random.uniform(lb, ub, (num_bats, dim))
velocities = np.zeros((num_bats, dim))
frequencies = np.zeros(num_bats)
loudness = A * np.ones(num_bats)
pulse_rate = r0 * np.ones(num_bats)

# Evaluate initial fitness
fitness = np.apply_along_axis(sphere_function, 1, positions)
best_position = positions[np.argmin(fitness)]
best_fitness = np.min(fitness)

for iteration in range(num_iterations):
    avg_loudness = np.mean(loudness)
    avg_pulse_rate = np.mean(pulse_rate)
    
    # Update bats
    for i in range(num_bats):
        beta = np.random.uniform(0, 1)
        frequencies[i] = freq_min + (freq_max - freq_min) * beta
        velocities[i] += (positions[i] - best_position) * frequencies[i]
        new_position = positions[i] + velocities[i]
        
        # Boundary check
        new_position = np.clip(new_position, lb, ub)
        
        # Local search
        if np.random.uniform(0, 1) > pulse_rate[i]:
            epsilon = np.random.uniform(-1, 1)
            new_position = positions[i] + epsilon * avg_loudness
        
        # Evaluate new solution
        new_fitness = sphere_function(new_position)
        
        # Greedy mechanism to update if new solution is better and random value is less than loudness
        if new_fitness < fitness[i] and np.random.uniform(0, 1) < loudness[i]:
            positions[i] = new_position
            fitness[i] = new_fitness
            
        # Update global best
        if fitness[i] < best_fitness:
            best_position = positions[i]
            best_fitness = fitness[i]
            
        loudness[i] *= alpha
        pulse_rate[i] = r0 * (1 - np.exp(-gamma * iteration))
            
    # Print the best fitness value in each iteration
    print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

print("\nOptimized Solution:", best_position)
print("Best Fitness Value:", best_fitness)