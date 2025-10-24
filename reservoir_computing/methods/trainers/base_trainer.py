import numpy as np
from abc import ABC, abstractmethod
from reservoir_computing.components.model import ReservoirComputingModel
from reservoir_computing.config_loader import ConfigLoader

class BaseTrainer(ABC):
    """
    Abstract base class for a Trainer in a Reservoir Computing model.
    Subclasses must implement the `run_training` method.
    """
    def __init__(self, model: ReservoirComputingModel, config_loader: ConfigLoader = None):
        self.model = model
        self.config_loader = config_loader if config_loader is not None else ConfigLoader()
        self.original_min_X = None
        self.original_max_X = None
        self.original_min_y = None
        self.original_max_y = None

    @abstractmethod
    def run_training(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Runs the training process for the specific trainer type.
        """
        pass
