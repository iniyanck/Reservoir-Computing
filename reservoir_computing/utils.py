import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes data to the range [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def denormalize_data(normalized_data: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """
    Denormalizes data from [0, 1] back to its original range.
    """
    return normalized_data * (original_max - original_min) + original_min

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    """
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
