import numpy as np

class Reservoir:
    """
    Abstract base class for a Reservoir in a Reservoir Computing model.
    Subclasses must implement the `_initialize_weights`, `_compute_state`, and `_update_state` methods.
    """
    def __init__(self, input_dim, reservoir_dim, spectral_radius, sparsity, random_state=None):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state = random_state if random_state is not None else np.random.RandomState()

        self.W_in = None  # Input weights
        self.W_res = None # Reservoir weights
        self.state = None # Current reservoir state

    def _initialize_weights(self):
        """
        Initializes input and reservoir weights. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _initialize_weights method.")

    def _compute_state(self, input_data, previous_state):
        """
        Computes the next reservoir state based on input and previous state. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compute_state method.")

    def _update_state(self, new_state):
        """
        Updates the internal reservoir state. Can be overridden by subclasses if needed.
        """
        self.state = new_state

    def run(self, input_sequence):
        """
        Runs the reservoir for a given input sequence and collects all states.
        """
        if self.state is None:
            self.state = np.zeros((1, self.reservoir_dim)) # Initialize state if not already set

        states = []
        for t in range(input_sequence.shape[0]):
            input_t = input_sequence[t:t+1, :] # Get current input
            new_state = self._compute_state(input_t, self.state)
            self._update_state(new_state)
            states.append(self.state.flatten())
        return np.array(states)

class RNNReservoir(Reservoir):
    """
    A simple RNN-based Reservoir implementation.
    """
    def __init__(self, input_dim, reservoir_dim, spectral_radius, sparsity, 
                 leaking_rate=1.0, input_scaling=1.0, activation=np.tanh, random_state=None):
        super().__init__(input_dim, reservoir_dim, spectral_radius, sparsity, random_state)
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.activation = activation
        self._initialize_weights() # Call initialize_weights after all attributes are set

    def _initialize_weights(self):
        # Input weights (W_in)
        # Scale W_in by input_scaling
        self.W_in = (self.random_state.rand(self.input_dim, self.reservoir_dim) * 2 - 1) * self.input_scaling

        # Reservoir weights (W_res)
        self.W_res = self.random_state.rand(self.reservoir_dim, self.reservoir_dim) * 2 - 1

        # Apply sparsity
        mask = self.random_state.rand(self.reservoir_dim, self.reservoir_dim) > self.sparsity
        self.W_res[mask] = 0

        # Scale reservoir weights to desired spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if radius > 0:
            self.W_res = self.W_res * (self.spectral_radius / radius)

    def _compute_state(self, input_data, previous_state):
        """
        Computes the next state for the RNN reservoir with leaky integration.
        x_tilde(t) = activation(W_in * input(t) + W_res * state(t-1))
        state(t) = (1 - alpha) * state(t-1) + alpha * x_tilde(t)
        """
        # Ensure input_data and previous_state have correct dimensions
        # input_data: (1, input_dim)
        # previous_state: (1, reservoir_dim)

        # Calculate the weighted sum for x_tilde
        weighted_sum = np.dot(input_data, self.W_in) + np.dot(previous_state, self.W_res)

        # Apply activation function to get x_tilde
        x_tilde = self.activation(weighted_sum)

        # Apply leaky integration
        new_state = (1 - self.leaking_rate) * previous_state + self.leaking_rate * x_tilde
        return new_state
