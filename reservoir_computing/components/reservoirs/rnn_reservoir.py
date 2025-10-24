import numpy as np
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.reservoirs.base_reservoir import Reservoir

class RNNReservoir(Reservoir):
    """
    A simple RNN-based Reservoir implementation with optional multiple layers.
    """
    def __init__(self, input_dim=None, reservoir_dim=None, spectral_radius=None, sparsity=None, 
                 leaking_rate=None, input_scaling=None, activation=np.tanh, num_layers=None, random_state=None, config_loader=None):
        
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        # Load parameters from config if not provided
        input_dim = input_dim if input_dim is not None else self.config_loader.get('reservoir.n_inputs')
        reservoir_dim = reservoir_dim if reservoir_dim is not None else self.config_loader.get('reservoir.n_reservoir')
        spectral_radius = spectral_radius if spectral_radius is not None else self.config_loader.get('reservoir.spectral_radius')
        sparsity = sparsity if sparsity is not None else self.config_loader.get('reservoir.sparsity')
        leaking_rate = leaking_rate if leaking_rate is not None else self.config_loader.get('reservoir.leaking_rate')
        input_scaling = input_scaling if input_scaling is not None else self.config_loader.get('reservoir.input_scaling')
        num_layers = num_layers if num_layers is not None else self.config_loader.get('reservoir.num_layers')
        random_state_seed = random_state if random_state is not None else self.config_loader.get('reservoir.random_state')
        self.random_state = np.random.RandomState(random_state_seed) # Initialize RandomState object

        super().__init__(input_dim, reservoir_dim, spectral_radius, sparsity, self.random_state)
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.activation = activation
        self.num_layers = num_layers
        self.W_res_layers = [] # To store weights for multiple layers
        self._initialize_weights() # Call initialize_weights after all attributes are set

    def _initialize_weights(self):
        # Input weights (W_in)
        # Scale W_in by input_scaling
        self.W_in = (self.random_state.rand(self.input_dim, self.reservoir_dim) * 2 - 1) * self.input_scaling

        # Reservoir weights (W_res) for each layer
        self.W_res_layers = []
        for _ in range(self.num_layers):
            W_res_layer = self.random_state.rand(self.reservoir_dim, self.reservoir_dim) * 2 - 1

            # Apply sparsity
            mask = self.random_state.rand(self.reservoir_dim, self.reservoir_dim) > self.sparsity
            W_res_layer[mask] = 0

            # Scale reservoir weights to desired spectral radius
            radius = np.max(np.abs(np.linalg.eigvals(W_res_layer)))
            if radius > 0:
                W_res_layer = W_res_layer * (self.spectral_radius / radius)
            self.W_res_layers.append(W_res_layer)

    def _compute_state(self, input_data, previous_state):
        """
        Computes the next state for the RNN reservoir with leaky integration and multiple layers.
        x_tilde(t) = activation(W_in * input(t) + W_res_layer_1 * state(t-1) + W_res_layer_2 * x_tilde_layer_1 + ...)
        state(t) = (1 - alpha) * state(t-1) + alpha * x_tilde(t)
        """
        # Ensure input_data and previous_state have correct dimensions
        # input_data: (1, input_dim)
        # previous_state: (1, reservoir_dim)

        # Initialize the state for the first layer's computation
        current_state_for_layer = previous_state

        # Compute the state for each layer
        for i, W_res_layer in enumerate(self.W_res_layers):
            if i == 0:
                # First layer receives input_data and previous_state
                weighted_sum = np.dot(input_data, self.W_in) + np.dot(current_state_for_layer, W_res_layer)
            else:
                # Subsequent layers receive the activated output of the previous layer
                weighted_sum = np.dot(current_state_for_layer, W_res_layer)
            
            current_state_for_layer = self.activation(weighted_sum)
        
        x_tilde = current_state_for_layer # The output of the last layer is x_tilde

        # Apply leaky integration
        new_state = (1 - self.leaking_rate) * previous_state + self.leaking_rate * x_tilde
        return new_state
