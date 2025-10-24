import numpy as np
from abc import ABC, abstractmethod

class Reservoir(ABC):
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

    @abstractmethod
    def _initialize_weights(self):
        """
        Initializes input and reservoir weights. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_state(self, input_data, previous_state):
        """
        Computes the next reservoir state based on input and previous state. Must be implemented by subclasses.
        """
        pass

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
