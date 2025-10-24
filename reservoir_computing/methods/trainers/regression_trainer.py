import numpy as np
from reservoir_computing.components.model import ReservoirComputingModel
from reservoir_computing.utils import mean_squared_error
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.methods.trainers.base_trainer import BaseTrainer

class RegressionTrainer(BaseTrainer):
    """
    A trainer for Reservoir Computing models using regression (e.g., Ridge Regression).
    """
    def __init__(self, model: ReservoirComputingModel, config_loader: ConfigLoader = None):
        super().__init__(model, config_loader)

    def run_training(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        regularization_coeff = self.config_loader.get('trainer.regularization', 0.0)
        
        print("Training model (Regression)...")
        self.model.train(X_train, y_train, regularization_coeff=regularization_coeff)
        print("Training complete.")

        print("Making predictions on training data...")
        predictions_train = self.model.predict(X_train)
        print("Making predictions on test data...")
        predictions_test = self.model.predict(X_test)

        mse_train = mean_squared_error(y_train, predictions_train)
        mse_test = mean_squared_error(y_test, predictions_test)

        print(f"Mean Squared Error (Train): {mse_train:.4f}")
        print(f"Mean Squared Error (Test): {mse_test:.4f}")

        return predictions_train, predictions_test, mse_train, mse_test
