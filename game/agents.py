import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)


# NEURAL NETWORK AGENT
# Define the structure of the neural network layers
def neuralNetworkLayers(grid_size = 5, useBias = True): 
    layers_shape = [
        (grid_size**2 + 2, 32),  # Primeira camada: 27 entradas, 32 neurônios
        (32, 16),  # Segunda camada: 32 entradas, 16 neurônios
        (16, 3)    # Camada de saída: 16 entradas, 3 saídas (ações)
    ]
    if not useBias:
        nodes = sum(layer_in * layer_out for layer_in, layer_out in layers_shape)
    else:
        nodes = sum(layer_in * layer_out + layer_out for layer_in, layer_out in layers_shape)

    return layers_shape, nodes

class NeuralNetworkAgent(Agent):
    def __init__(self, grid_size: int, useBias =True, network_setup: np.ndarray):
        """
        Initialize the neural network agent with weights.
        
        Args:
            network_setup: Linear array of weights and biases for each layer of the neural network.
        """
        self.useBias = useBias

        layers, nodes_len = neuralNetworkLayers(grid_size, useBias)
        if len(network_setup) != nodes_len:
            raise ValueError("Invalid network setup length.")

        # Transform linear input into bias and weights arrays
        weights = []
        idx = 0
        for layer_in, layer_out in layers:
            w_size = layer_in * layer_out
            b_size = layer_out
            weights.append(np.array(network_setup[idx: idx + w_size]).reshape(layer_in, layer_out))
            idx += w_size
            if useBias:
                weights.append(np.array(network_setup[idx: idx + b_size]).reshape(layer_out))
                idx += b_size

        if (useBias and len(weights) != 6) or (not useBias and len(weights) != 3):
            raise ValueError("Expected 6 weight arrays: [W1, b1, W2, b2, W3, b3]")
        
        # Validate weight dimensions
        if weights[0].shape != (27, 32):
            raise ValueError(f"First layer weights should be (27, 32), got {weights[0].shape}")
        if weights[1].shape != (32, 16):
            raise ValueError(f"Second layer weights should be (32, 16), got {weights[1].shape}")
        if weights[2].shape != (16, 3):
            raise ValueError(f"Output layer weights should be (16, 3), got {weights[2].shape}")
        if weights[3].shape != (32,):
            raise ValueError(f"First layer bias should be (32,), got {weights[1].shape}")
        if weights[4].shape != (16,):
            raise ValueError(f"Second layer bias should be (16,), got {weights[3].shape}")
        if weights[5].shape != (3,):
            raise ValueError(f"Output layer bias should be (3,), got {weights[5].shape}")
        
        self.W1 = weights[0]
        self.W2 = weights[1]
        self.W3 = weights[2]
        self.b1 = weights[3]
        self.b2 = weights[4]
        self.b3 = weights[5]
    
    def predict(self, state: np.ndarray) -> int:
        """
        Forward pass through the neural network.
        
        Args:
            state: Input state vector of length 27
            
        Returns:
            int: Action index (0: up, 1: down, 2: noop)
        """
        if len(state) != 27:
            raise ValueError(f"Expected state of length 27, got {len(state)}")
        
        # Forward pass
        # First hidden layer with tanh activation
        z1 = np.dot(state, self.W1) 
        if self.useBias: z1 += self.b1
        a1 = np.tanh(z1)
        
        # Second hidden layer with tanh activation
        z2 = np.dot(a1, self.W2) 
        if self.useBias: z2 += self.b2
        a2 = np.tanh(z2)
        
        # Output layer (no activation, raw logits)
        z3 = np.dot(a2, self.W3)
        if self.useBias: z3 += self.b3

        # Return the action with highest output value
        return np.argmax(z3)
