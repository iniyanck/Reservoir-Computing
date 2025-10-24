import torch
import numpy as np
from reservoir_computing.components.reservoirs.base_reservoir import Reservoir

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
        self.W_out = None # Readout weights, will be a torch.Tensor

    def to(self, device):
        """
        Moves the model's readout weights to the specified device.
        """
        self.reservoir.to(device) # Move reservoir to device
        if self.W_out is not None:
            self.W_out = self.W_out.to(device)
        return self

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
        all_states_np = self.reservoir.run(input_sequence)
        all_states = torch.tensor(all_states_np, dtype=torch.float32).to(self.reservoir.device)
        target_sequence_t = torch.tensor(target_sequence, dtype=torch.float32).to(self.reservoir.device)

        # Apply washout
        if self.washout_steps > 0:
            states_for_training = all_states[self.washout_steps:]
            targets_for_training = target_sequence_t[self.washout_steps:]
        else:
            states_for_training = all_states
            targets_for_training = target_sequence_t

        # Add bias term to states
        bias_term = torch.ones((states_for_training.shape[0], 1), device=self.reservoir.device)
        states_with_bias = torch.cat([states_for_training, bias_term], dim=1)

        # Compute readout weights using ridge regression
        # (X^T @ X + beta * I) @ W_out = X^T @ Y_target
        A = torch.matmul(states_with_bias.T, states_with_bias) + regularization_coeff * torch.eye(states_with_bias.shape[1], device=self.reservoir.device)
        b = torch.matmul(states_with_bias.T, targets_for_training)
        self.W_out = torch.linalg.solve(A, b)

    def update_readout_weights_rl(self, state_with_bias: np.ndarray, target: np.ndarray, prediction: np.ndarray, learning_rate: float):
        """
        Updates the readout weights (W_out) using a simple online learning rule
        based on the prediction error. This is suitable for continuous RL.

        Args:
            state_with_bias (np.ndarray): The current reservoir state with bias, shape (1, reservoir_dim + 1).
            target (np.ndarray): The true target value for the current timestep, shape (1, output_dim).
            prediction (np.ndarray): The model's prediction for the current timestep, shape (1, output_dim).
            learning_rate (float): The learning rate for the weight update.
        """
        # Convert inputs to torch tensors and move to device
        state_with_bias_t = torch.tensor(state_with_bias, dtype=torch.float32).to(self.reservoir.device)
        target_t = torch.tensor(target, dtype=torch.float32).to(self.reservoir.device)
        prediction_t = torch.tensor(prediction, dtype=torch.float32).to(self.reservoir.device)
        
        if self.W_out is None:
            # Initialize W_out if it hasn't been trained yet (e.g., with zeros or small random values)
            self.W_out = torch.zeros((state_with_bias_t.shape[1], self.output_dim), device=self.reservoir.device)

        error = target_t - prediction_t # Shape (1, output_dim)
        
        # Simple delta rule / gradient descent update
        self.W_out += learning_rate * torch.matmul(state_with_bias_t.T, error)

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
        all_states_np = self.reservoir.run(input_sequence)
        all_states = torch.tensor(all_states_np, dtype=torch.float32).to(self.reservoir.device)

        # Add bias term to states for prediction
        bias_term = torch.ones((all_states.shape[0], 1), device=self.reservoir.device)
        states_with_bias = torch.cat([all_states, bias_term], dim=1)

        # Compute output
        predictions = torch.matmul(states_with_bias, self.W_out)
        return predictions.cpu().numpy()

    def free_run_predict(self, initial_input_sequence: np.ndarray, prediction_steps: int, true_input_for_prediction: np.ndarray = None) -> np.ndarray:
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
        
        # The reservoir's state is already a torch tensor on the correct device due to base_reservoir.py changes
        # if self.reservoir.state is None:
        #     self.reservoir.state = torch.zeros((1, self.reservoir.reservoir_dim), device=self.reservoir.device)

        # 1. Prime the reservoir with the initial_input_sequence
        # This updates the internal state of the reservoir
        initial_input_sequence_t = torch.tensor(initial_input_sequence, dtype=torch.float32).to(self.reservoir.device)
        for t in range(initial_input_sequence_t.shape[0]):
            input_t = initial_input_sequence_t[t:t+1, :] # Shape (1, input_dim)
            new_state = self.reservoir._compute_state(input_t, self.reservoir.state)
            self.reservoir._update_state(new_state)

        # Get the last state after priming
        current_reservoir_state = self.reservoir.state.clone() # Shape (1, reservoir_dim)

        # Make an initial prediction based on the last primed state
        bias_term_initial = torch.ones((1, 1), device=self.reservoir.device)
        last_primed_state_with_bias = torch.cat([current_reservoir_state, bias_term_initial], dim=1)
        current_prediction = torch.matmul(last_primed_state_with_bias, self.W_out) # Shape (1, output_dim)

        free_run_predictions = []
        free_run_predictions.append(current_prediction.flatten())

        # 2. Free-running prediction
        for i in range(prediction_steps - 1): # -1 because we already made one prediction
            # For double pendulum, input is (x1, y1, x2, y2)
            # We use true (x1, y1) and predicted (x2, y2)
            # Default behavior for free-running: previous prediction becomes the input for the next step.
            # If the model is predicting all input dimensions, this is correct.
            # If only a subset of inputs are predicted, more complex logic would be needed here
            # to combine true inputs with predicted outputs.
            input_for_next_step = current_prediction # Shape (1, output_dim)

            # Update reservoir state using the constructed input
            new_reservoir_state = self.reservoir._compute_state(input_for_next_step, current_reservoir_state)
            self.reservoir._update_state(new_reservoir_state) # Update internal state
            current_reservoir_state = new_reservoir_state.clone() # Keep track of the state

            # Make a new prediction based on the new reservoir state
            new_state_with_bias = torch.cat([current_reservoir_state, torch.ones((1, 1), device=self.reservoir.device)], dim=1)
            current_prediction = torch.matmul(new_state_with_bias, self.W_out)
            free_run_predictions.append(current_prediction.flatten())

        return torch.stack(free_run_predictions).cpu().numpy()
