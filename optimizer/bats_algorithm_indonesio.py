import numpy as np
import math

# np.random.seed(100)

class BatAlgorithm():
    def __init__(self, dimensions, n_bat=100, n_gen=1000, r0=0.9, A0=0.9, alpha=0.9, gamma=0.9, fmin=0.0, fmax=1.0, lower_bound=-1.0, upper_bound=1.0, function=None):
        self.dimensions = dimensions
        self.n_bat = n_bat
        self.n_gen = n_gen
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.function = function
        self.r0 = r0
        self.A0 = A0

        # If all bats are initialized with the same loudness and pulse rate,
        # we can use a single value for all bats
        self.A = self.A0  # Loudness
        self.r = self.r0  # Pulse rate
        
        # Initialize frequencies at zero
        self.f = np.zeros(n_bat)
        
        # Initialize with 0 velocity for all bats
        self.v = np.ndarray((n_bat, self.dimensions))
        self.v.fill(0.0)
        
        # Initialize random position for all bats
        self.x = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(n_bat, self.dimensions))
        
        # Run fitness, initialize best fitness and solution
        self.fitness = function(self.x)
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.x[best_idx]


    def optimize(self):
        avg_dist = 0.0
        for n in range(self.n_gen):
            print(f"Gen {n} Best fitness {self.best_fitness}")

            avg_loudness = np.mean(self.A)
            
            # Update frquency
            random = np.random.uniform(0,1, (self.n_bat,))
            self.f = self.fmin + (self.fmax-self.fmin)*random
            
            # Update velocity
            self.v += (self.x - self.best_solution) * self.f[:, np.newaxis]
            
            # Update position
            new_solution = np.clip(self.x + self.v, self.lower_bound, self.upper_bound)
                
            for i in range(self.n_bat):
                for j in range(i, self.n_bat):
                    avg_dist += np.linalg.norm(np.array(self.x[i]) - np.array(self.x[j]))
                # Search if randomly bigger than pulse_rate
                random = np.random.uniform(0,1)
                if(random > self.r):
                    random = np.random.uniform(-1.0,1.0)
                    # TODO try use local search here
                    new_solution[i] = np.clip(self.best_solution + random * avg_loudness, self.lower_bound, self.upper_bound)  

            avg_dist /= self.n_bat
            print(f"Gen {n} Average distance: {avg_dist}")

            # Test new solutions
            new_fitness = self.function(new_solution)

            for i in range(self.n_bat):
                # Accept new solution if it is better and random is less than loudness
                random = np.random.uniform(0,1)

                # Alteração no critério de aceitação
                # Always accept new solution if it is better than current 
                # Accept worse if random is less than loudness
                if new_fitness[i] > self.fitness[i] or (random < self.A):
                    self.x[i] = new_solution[i]
                    self.fitness[i] = new_fitness[i]

                if self.fitness[i] > self.best_fitness:
                    self.best_fitness = new_fitness[i]
                    self.best_solution = self.x[i]
                    
                # Update loudness and pulse rate
                self.A = self.A*self.alpha
                self.r = self.r0*(1 - math.exp(-1*self.gamma*n))
        # Print the best solution found
        # print("Best solution ",self.best_solution)
        print("Best fitness ",self.best_fitness)