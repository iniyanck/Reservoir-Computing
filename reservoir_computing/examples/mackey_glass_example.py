import numpy as np
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.example import Example

class MackeyGlassExample(Example):
    """
    A class to generate Mackey-Glass time series data.
    """
    def __init__(self, config_loader: ConfigLoader):
        super().__init__()
        self.config_loader = config_loader
        self.n_samples = self.config_loader.get('mackey_glass.n_samples', 2000)
        self.tau = self.config_loader.get('mackey_glass.tau', 17)
        self.delay = self.config_loader.get('mackey_glass.delay', 100)
        self.seed = self.config_loader.get('mackey_glass.seed', 42)
        
        self.time_steps = np.arange(self.n_samples) # For plotting

    def generate_data(self):
        """
        Generates a Mackey-Glass time series.
        Returns:
            np.ndarray: The generated Mackey-Glass time series.
        """
        np.random.seed(self.seed)
        x = np.zeros(self.n_samples + self.delay)
        x[0:self.delay] = 0.5 + 0.1 * np.random.rand(self.delay) # Initial conditions
        
        a = 0.2
        b = 0.1
        gamma = 1.0
        n = 10

        for i in range(self.delay, self.n_samples + self.delay -1):
            x_tau = x[i - self.tau]
            x[i] = x[i-1] + (a * x_tau / (1 + x_tau**n) - b * x[i-1]) / gamma
        return x[self.delay:].reshape(-1, 1)
