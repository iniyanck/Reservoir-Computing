import numpy as np
from .reservoir import Reservoir

class ReservoirComputingModel:
    """
    A generic Reservoir Computing model that combines a reservoir with a linear readout.
    """
    def __init__(self, reservoir: Reservoir, output_dim: int, washout_steps: int = 0):
        if not isinstance(reservoir, Reservoir):
            raise TypeError("reservoir must be an instance of Reservoir or its subclass.")

        self.reservoir = reservoir
        self.output_dim = output_dim
        self.washout_steps = washout_steps
        self.W_out = None # Readout weights

    def train(self, input_sequence: np.ndarray, target_sequence: np.ndarray, regularization_coeff: float = 0.0):
        """
        Trains the readout layer using a given input and target sequence.
        The readout weights (W_out) are computed using linear regression (ridge regression).

        Args:
            input_sequence (np.ndarray): The input data to the reservoir. Shape (timesteps, input_dim).
            target_sequence (np.ndarray): The target output data. Shape (timesteps, output_dim).
            regularization_coeff (float): The regularization coefficient (beta) for ridge regression.
                                          If 0, it defaults to standard linear regression (pseudo-inverse).
        """
        if input_sequence.shape[0] != target_sequence.shape[0]:
            raise ValueError("Input and target sequences must have the same number of timesteps.")

        # Run the reservoir to collect states
        all_states = self.reservoir.run(input_sequence)

        # Apply washout
        if self.washout_steps > 0:
            states_for_training = all_states[self.washout_steps:]
            targets_for_training = target_sequence[self.washout_steps:]
        else:
            states_for_training = all_states
            targets_for_training = target_sequence

        # Add bias term to states
        states_with_bias = np.hstack([states_for_training, np.ones((states_for_training.shape[0], 1))])

        # Compute readout weights using ridge regression
        # W_out = Y_target * X^T * (X * X^T + beta * I)^-1
        # Or, in the form (X^T * X + beta * I)^-1 * X^T * Y_target
        
        # X_T_X = states_with_bias.T @ states_with_bias
        # identity_matrix = np.eye(X_T_X.shape[0])
        # self.W_out = np.linalg.inv(X_T_X + regularization_coeff * identity_matrix) @ states_with_bias.T @ targets_for_training

        # A more numerically stable way using np.linalg.solve for (A @ x = b)
        # (X^T @ X + beta * I) @ W_out = X^T @ Y_target
        A = states_with_bias.T @ states_with_bias + regularization_coeff * np.eye(states_with_bias.shape[1])
        b = states_with_bias.T @ targets_for_training
        self.W_out = np.linalg.solve(A, b)

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Predicts output for a given input sequence using the trained model.

        Args:
            input_sequence (np.ndarray): The input data to the reservoir. Shape (timesteps, input_dim).

        Returns:
            np.ndarray: The predicted output sequence. Shape (timesteps, output_dim).
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been trained. Call 'train' method first.")

        # Run the reservoir to collect states
        all_states = self.reservoir.run(input_sequence)

        # Add bias term to states for prediction
        states_with_bias = np.hstack([all_states, np.ones((all_states.shape[0], 1))])

        # Compute output
        predictions = np.dot(states_with_bias, self.W_out)
        return predictions

    def free_run_predict(self, initial_input_sequence: np.ndarray, prediction_steps: int) -> np.ndarray:
        """
        Generates predictions in a free-running (generative) mode.
        The model's own previous output is fed back as the input for the next step.

        Args:
            initial_input_sequence (np.ndarray): An initial input sequence to prime the reservoir.
                                                 Shape (priming_timesteps, input_dim).
            prediction_steps (int): The number of steps to predict in free-running mode.

        Returns:
            np.ndarray: The predicted output sequence in free-running mode.
                        Shape (prediction_steps, output_dim).
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been trained. Call 'train' method first.")
        if self.reservoir.state is None:
            # Initialize reservoir state if not already set (e.g., if train was not called)
            self.reservoir.state = np.zeros((1, self.reservoir.reservoir_dim))

        # 1. Prime the reservoir with the initial_input_sequence
        # This updates the internal state of the reservoir
        for t in range(initial_input_sequence.shape[0]):
            input_t = initial_input_sequence[t:t+1, :] # Shape (1, input_dim)
            new_state = self.reservoir._compute_state(input_t, self.reservoir.state)
            self.reservoir._update_state(new_state)

        # Get the last state after priming
        current_reservoir_state = self.reservoir.state.copy() # Shape (1, reservoir_dim)

        # Make an initial prediction based on the last primed state
        last_primed_state_with_bias = np.hstack([current_reservoir_state, np.ones((1, 1))])
        current_prediction = np.dot(last_primed_state_with_bias, self.W_out) # Shape (1, output_dim)

        free_run_predictions = []
        free_run_predictions.append(current_prediction.flatten())

        # 2. Free-running prediction
        for _ in range(prediction_steps - 1): # -1 because we already made one prediction
            # The previous prediction becomes the input for the next step
            input_for_next_step = current_prediction # Shape (1, output_dim)

            # Update reservoir state using the predicted output as input
            new_reservoir_state = self.reservoir._compute_state(input_for_next_step, current_reservoir_state)
            self.reservoir._update_state(new_reservoir_state) # Update internal state
            current_reservoir_state = new_reservoir_state.copy() # Keep track of the state

            # Make a new prediction based on the new reservoir state
            new_state_with_bias = np.hstack([current_reservoir_state, np.ones((1, 1))])
            current_prediction = np.dot(new_state_with_bias, self.W_out)
            free_run_predictions.append(current_prediction.flatten())

        return np.array(free_run_predictions)
