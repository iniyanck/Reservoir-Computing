import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.example import Example # Import the abstract base class

class DoublePendulum(Example): # Inherit from Example
    def __init__(self, config_loader: ConfigLoader = None):
        super().__init__() # Call the base class constructor
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        self.L1 = self.config_loader.get('double_pendulum.l1', 1.0)
        self.L2 = self.config_loader.get('double_pendulum.l2', 1.0)
        self.M1 = self.config_loader.get('double_pendulum.m1', 1.0)
        self.M2 = self.config_loader.get('double_pendulum.m2', 1.0)
        self.G = self.config_loader.get('double_pendulum.g', 9.81) # Gravity constant
        self.timesteps = self.config_loader.get('double_pendulum.timesteps', 5000)
        self.dt = self.config_loader.get('double_pendulum.dt', 0.01)
        self.initial_state = [np.pi/2, 0, np.pi/2, 0] # Default initial angles and velocities

        self.time_steps = np.arange(0, self.timesteps * self.dt, self.dt)
        if len(self.time_steps) > self.timesteps:
            self.time_steps = self.time_steps[:self.timesteps]
        elif len(self.time_steps) < self.timesteps:
            # Adjust dt slightly if needed to hit exact timesteps, or pad
            # For simplicity, we'll just ensure it's the correct length
            self.time_steps = np.linspace(0, (self.timesteps - 1) * self.dt, self.timesteps)


    def _deriv(self, state, t):
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (self.L1 * (self.M1 + self.M2) - self.M2 * self.L1 * np.cos(delta)**2)
        dydx[1] = (self.M2 * self.L1 * state[1]**2 * np.sin(delta) * np.cos(delta) +
                   self.M2 * self.G * np.sin(state[2]) * np.cos(delta) +
                   self.M2 * self.L2 * state[3]**2 * np.sin(delta) -
                   (self.M1 + self.M2) * self.G * np.sin(state[0])) / den1

        dydx[2] = state[3]

        den2 = (self.L2 * (self.M1 + self.M2) - self.M2 * self.L2 * np.cos(delta)**2)
        dydx[3] = (-self.M2 * self.L2 * state[3]**2 * np.sin(delta) * np.cos(delta) +
                   (self.M1 + self.M2) * self.G * np.sin(state[0]) * np.cos(delta) -
                   (self.M1 + self.M2) * self.L1 * state[1]**2 * np.sin(delta) -
                   (self.M1 + self.M2) * self.G * np.sin(state[2])) / den2
        return dydx

    def simulate(self, initial_state, t_points):
        states = odeint(self._deriv, initial_state, t_points)

        theta1 = states[:, 0]
        theta2 = states[:, 2]

        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)

        x2 = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta2)
        y2 = -self.L1 * np.cos(theta1) - self.L2 * np.cos(theta2)

        return states, x1, y1, x2, y2

    def generate_data(self):
        """
        Generates double pendulum (x,y) movement data.
        Returns:
            np.ndarray: Stacked input and target data.
                        Input: (x1, y1, x2, y2) at time t
                        Target: (x2, y2) at time t+1 (or just (x2, y2) at time t for prediction)
        """
        np.random.seed(42) # Ensure reproducibility

        states, x1, y1, x2, y2 = self.simulate(self.initial_state, self.time_steps)
        
        # The model will predict (x2, y2) based on (x1, y1, x2, y2) at the current timestep.
        # So, the input to the model is (x1, y1, x2, y2) and the target is (x2, y2) of the next step.
        
        # For simplicity, let's make the model predict (x2, y2) at the current timestep
        # based on (x1, y1, x2, y2) at the current timestep.
        # This means the input and target are derived from the same time step.
        
        # Input to the model: (x1, y1, x2, y2)
        input_data = np.vstack((x1, y1, x2, y2)).T # Shape (timesteps, 4)
        
        # Target for the model: (x1, y1, x2, y2)
        target_data = np.vstack((x1, y1, x2, y2)).T # Shape (timesteps, 4)

        # The main.py expects a single data array. We'll stack inputs and targets.
        # The trainer will then split them.
        # So, we return a combined array where the first 4 columns are input and last 2 are target.
        return np.hstack((input_data, target_data))


def animate_double_pendulum(true_x1, true_y1, true_x2, true_y2,
                            predicted_x1, predicted_y1, predicted_x2, predicted_y2,
                            L1=1.0, L2=1.0, interval=20):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_title('Double Pendulum Simulation: True vs Predicted')
    ax.grid(True)

    line_true, = ax.plot([], [], 'o-', lw=2, color='blue', label='True Pendulum')
    time_text_true = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='blue')

    line_pred, = ax.plot([], [], 'o-', lw=2, color='red', label='Predicted Pendulum')
    time_text_pred = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='red')

    ax.legend()

    def init():
        line_true.set_data([], [])
        time_text_true.set_text('')
        line_pred.set_data([], [])
        time_text_pred.set_text('')
        return line_true, time_text_true, line_pred, time_text_pred

    def update(frame):
        x_true = [0, true_x1[frame], true_x2[frame]]
        y_true = [0, true_y1[frame], true_y2[frame]]
        line_true.set_data(x_true, y_true)
        time_text_true.set_text(f'True Time: {frame * interval / 1000:.2f}s')

        x_pred = [0, predicted_x1[frame], predicted_x2[frame]]
        y_pred = [0, predicted_y1[frame], predicted_y2[frame]]
        line_pred.set_data(x_pred, y_pred)
        time_text_pred.set_text(f'Pred Time: {frame * interval / 1000:.2f}s')

        return line_true, time_text_true, line_pred, time_text_pred

    ani = FuncAnimation(fig, update, frames=len(true_x2),
                        init_func=init, blit=True, interval=interval)
    return ani

def animate_free_run_pendulum(predicted_x1, predicted_y1, predicted_x2, predicted_y2,
                              L1=1.0, L2=1.0, interval=20):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_title('Double Pendulum Simulation: Free-Run Prediction')
    ax.grid(True)

    line_pred, = ax.plot([], [], 'o-', lw=2, color='red', label='Free-Run Predicted Pendulum')
    time_text_pred = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='red')

    ax.legend()

    def init():
        line_pred.set_data([], [])
        time_text_pred.set_text('')
        return line_pred, time_text_pred

    def update(frame):
        x_pred = [0, predicted_x1[frame], predicted_x2[frame]]
        y_pred = [0, predicted_y1[frame], predicted_y2[frame]]
        line_pred.set_data(x_pred, y_pred)
        time_text_pred.set_text(f'Pred Time: {frame * interval / 1000:.2f}s')

        return line_pred, time_text_pred

    ani = FuncAnimation(fig, update, frames=len(predicted_x2),
                        init_func=init, blit=True, interval=interval)
    return ani
