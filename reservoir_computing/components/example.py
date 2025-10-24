from abc import ABC, abstractmethod
import numpy as np

class Example(ABC):
    """
    Abstract base class for all example problems.
    Each concrete example problem must implement the generate_data method.
    """
    def __init__(self):
        self.time_steps = None # To be set by concrete implementations

    @abstractmethod
    def generate_data(self) -> np.ndarray:
        """
        Generates the data for the specific example problem.
        Returns:
            np.ndarray: The generated data.
        """
        pass
