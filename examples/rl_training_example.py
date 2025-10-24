import numpy as np
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.example import Example

class RLTrainingExample(Example):
    """
    A class to generate data for RL training examples, typically a sine wave.
    """
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.timesteps = self.config_loader.get('rl_training.timesteps', 1000)
        self.frequency = self.config_loader.get('rl_training.frequency', 0.02)
        self.amplitude = self.config_loader.get('rl_training.amplitude', 1.0)
        self.noise_std = self.config_loader.get('rl_training.noise_std', 0.05)
        self.time_steps = np.arange(self.timesteps) # For plotting

    def generate_data(self):
        """Generates a sine wave with optional noise for RL training."""
        data = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time_steps)
        if self.noise_std > 0:
            data += np.random.normal(0, self.noise_std, self.timesteps)
        return data.reshape(-1, 1) # Ensure shape (timesteps, 1)

# The main execution logic is now handled by main.py
