import numpy as np
from reservoir_computing.components.model import ReservoirComputingModel
from reservoir_computing.utils import mean_squared_error
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.methods.trainers.base_trainer import BaseTrainer

class RLTrainer(BaseTrainer):
    """
    A trainer for Reservoir Computing models using an RL-like continuous training approach.
    """
    def __init__(self, model: ReservoirComputingModel, config_loader: ConfigLoader = None):
        super().__init__(model, config_loader)

    def run_training(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        rl_learning_rate = self.config_loader.get('trainer.rl_learning_rate', 0.01)
        rl_epochs = self.config_loader.get('trainer.rl_epochs', 1)
        
        print(f"Starting RL continuous training for {rl_epochs} epochs...")

        # Ensure reservoir state is initialized
        if self.model.reservoir.state is None:
            self.model.reservoir.state = np.zeros((1, self.model.reservoir.reservoir_dim))
        
        # Initialize W_out if it's None (e.g., if not pre-trained with batch method)
        if self.model.W_out is None:
            output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
            state_with_bias_dim = self.model.reservoir.reservoir_dim + 1
            self.model.W_out = np.zeros((state_with_bias_dim, output_dim))

        for epoch in range(rl_epochs):
            epoch_predictions = []
            epoch_targets = []
            
            self.model.reservoir.state = np.zeros((1, self.model.reservoir.reservoir_dim))

            for t in range(X_train.shape[0]):
                input_t = X_train[t:t+1, :]
                target_t = y_train[t:t+1, :]

                current_reservoir_state = self.model.reservoir.run(input_t)[-1:]
                
                state_with_bias = np.hstack([current_reservoir_state, np.ones((1, 1))])
                prediction_t = np.dot(state_with_bias, self.model.W_out)

                self.model.update_readout_weights_rl(state_with_bias, target_t, prediction_t, rl_learning_rate)

                epoch_predictions.append(prediction_t.flatten())
                epoch_targets.append(target_t.flatten())

            epoch_predictions_arr = np.array(epoch_predictions)
            epoch_targets_arr = np.array(epoch_targets)

            mse_epoch = mean_squared_error(epoch_targets_arr, epoch_predictions_arr)
            print(f"Epoch {epoch+1}/{rl_epochs}, MSE: {mse_epoch:.4f}")

        print("RL continuous training complete.")
        final_predictions_train = self.model.predict(X_train)
        final_mse_train = mean_squared_error(y_train, final_predictions_train)
        print(f"Final Mean Squared Error (Train) after RL training: {final_mse_train:.4f}")

        print("Making predictions on test data after RL training...")
        predictions_test = self.model.predict(X_test)
        mse_test = mean_squared_error(y_test, predictions_test)
        print(f"Mean Squared Error (Test) after RL training: {mse_test:.4f}")

        return final_predictions_train, predictions_test, final_mse_train, mse_test
