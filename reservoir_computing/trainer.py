import numpy as np
from .model import ReservoirComputingModel
from .utils import mean_squared_error, normalize_data, denormalize_data

class Trainer:
    """
    A trainer class for Reservoir Computing models, handling data preparation,
    training, and evaluation.
    """
    def __init__(self, model: ReservoirComputingModel):
        self.model = model
        self.original_min = None
        self.original_max = None

    def prepare_data(self, data: np.ndarray, train_ratio: float = 0.7, normalize: bool = True):
        """
        Prepares data for training and testing, including optional normalization.

        Args:
            data (np.ndarray): The full dataset. Shape (timesteps, features).
            train_ratio (float): The ratio of data to use for training.
            normalize (bool): Whether to normalize the data to [0, 1].

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if normalize:
            self.original_min = np.min(data)
            self.original_max = np.max(data)
            data = normalize_data(data)

        num_timesteps = data.shape[0]
        num_train = int(num_timesteps * train_ratio)

        # For time series prediction, input X(t) predicts output y(t+1)
        X = data[:-1]
        y = data[1:]

        X_train, y_train = X[:num_train], y[:num_train]
        X_test, y_test = X[num_train:], y[num_train:]

        return X_train, y_train, X_test, y_test

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, regularization_coeff: float = 0.0):
        """
        Trains the model and evaluates its performance on the test set.

        Args:
            X_train, y_train, X_test, y_test: Prepared data.
            regularization_coeff (float): The regularization coefficient for ridge regression.

        Returns:
            tuple: (predictions_train, predictions_test, mse_train, mse_test)
        """
        print("Training model...")
        self.model.train(X_train, y_train, regularization_coeff=regularization_coeff)
        print("Training complete.")

        print("Making predictions on training data...")
        predictions_train = self.model.predict(X_train)
        print("Making predictions on test data...")
        predictions_test = self.model.predict(X_test)

        if self.original_min is not None and self.original_max is not None:
            predictions_train = denormalize_data(predictions_train, self.original_min, self.original_max)
            predictions_test = denormalize_data(predictions_test, self.original_min, self.original_max)
            y_train = denormalize_data(y_train, self.original_min, self.original_max)
            y_test = denormalize_data(y_test, self.original_min, self.original_max)

        mse_train = mean_squared_error(y_train, predictions_train)
        mse_test = mean_squared_error(y_test, predictions_test)

        print(f"Mean Squared Error (Train): {mse_train:.4f}")
        print(f"Mean Squared Error (Test): {mse_test:.4f}")

        return predictions_train, predictions_test, mse_train, mse_test
