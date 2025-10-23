import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from .config_loader import ConfigLoader

class DoublePendulum:
    def __init__(self, L1=None, L2=None, M1=None, M2=None, config_loader=None):
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        self.L1 = L1 if L1 is not None else self.config_loader.get('double_pendulum.l1', 1.0)
        self.L2 = L2 if L2 is not None else self.config_loader.get('double_pendulum.l2', 1.0)
        self.M1 = M1 if M1 is not None else self.config_loader.get('double_pendulum.m1', 1.0)
        self.M2 = M2 if M2 is not None else self.config_loader.get('double_pendulum.m2', 1.0)
        self.G = self.config_loader.get('double_pendulum.g', 9.81) # Gravity constant

    def _deriv(self, state, t):
        """
        Returns the derivatives of the double pendulum state.
        state: [theta1, z1, theta2, z2]
               theta1: angle of pendulum 1
               z1: angular velocity of pendulum 1
               theta2: angle of pendulum 2
               z2: angular velocity of pendulum 2
        """
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
        """
        Simulates the double pendulum motion.
        initial_state: [theta1_0, z1_0, theta2_0, z2_0]
        t_points: array of time points
        Returns:
            states: array of states at each time point
            x1, y1: (x,y) coordinates of pendulum 1
            x2, y2: (x,y) coordinates of pendulum 2
        """
        states = odeint(self._deriv, initial_state, t_points)

        theta1 = states[:, 0]
        theta2 = states[:, 2]

        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)

        x2 = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta2)
        y2 = -self.L1 * np.cos(theta1) - self.L2 * np.cos(theta2)

        return states, x1, y1, x2, y2

def generate_double_pendulum_data(n_samples, dt=0.02, initial_state=None, L1=1.0, L2=1.0, M1=1.0, M2=1.0):
    """
    Generates double pendulum (x,y) movement data.
    """
    if initial_state is None:
        initial_state = [np.pi/2, 0, np.pi/2, 0] # [theta1, z1, theta2, z2]

    dp = DoublePendulum(L1, L2, M1, M2)
    t_points = np.arange(0, n_samples * dt, dt)
    
    if len(t_points) < n_samples: # Ensure we have exactly n_samples points
        t_points = np.arange(0, n_samples * dt, dt)[:n_samples]
    elif len(t_points) > n_samples:
        t_points = t_points[:n_samples]

    states, x1, y1, x2, y2 = dp.simulate(initial_state, t_points)

    # We want to predict (x2, y2) movement
    return x1, y1, x2, y2

from matplotlib.animation import FuncAnimation
from reservoir_computing.reservoir import RNNReservoir
from reservoir_computing.model import ReservoirComputingModel
from reservoir_computing.trainer import Trainer

def animate_double_pendulum(true_x1, true_y1, true_x2, true_y2,
                            predicted_x1, predicted_y1, predicted_x2, predicted_y2,
                            L1=1.0, L2=1.0, interval=20):
    """
    Animates the true and predicted double pendulum movement.
    true_x1, true_y1: (N,) arrays of true (x,y) coordinates of pendulum 1
    true_x2, true_y2: (N,) arrays of true (x,y) coordinates of pendulum 2
    predicted_x1, predicted_y1: (N,) arrays of predicted (x,y) coordinates of pendulum 1
    predicted_x2, predicted_y2: (N,) arrays of predicted (x,y) coordinates of pendulum 2
    L1, L2: lengths of the pendulum rods
    interval: delay between frames in ms
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_title('Double Pendulum Simulation: True vs Predicted')
    ax.grid(True)

    # True pendulum
    line_true, = ax.plot([], [], 'o-', lw=2, color='blue', label='True Pendulum')
    time_text_true = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='blue')

    # Predicted pendulum
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
        # True pendulum
        x_true = [0, true_x1[frame], true_x2[frame]]
        y_true = [0, true_y1[frame], true_y2[frame]]
        line_true.set_data(x_true, y_true)
        time_text_true.set_text(f'True Time: {frame * interval / 1000:.2f}s')

        # Predicted pendulum
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
    """
    Animates only the free-running predicted double pendulum movement.
    predicted_x1, predicted_y1: (N,) arrays of predicted (x,y) coordinates of pendulum 1
    predicted_x2, predicted_y2: (N,) arrays of predicted (x,y) coordinates of pendulum 2
    L1, L2: lengths of the pendulum rods
    interval: delay between frames in ms
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_title('Double Pendulum Simulation: Free-Run Prediction')
    ax.grid(True)

    # Predicted pendulum
    line_pred, = ax.plot([], [], 'o-', lw=2, color='red', label='Free-Run Predicted Pendulum')
    time_text_pred = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='red')

    ax.legend()

    def init():
        line_pred.set_data([], [])
        time_text_pred.set_text('')
        return line_pred, time_text_pred

    def update(frame):
        # Predicted pendulum
        x_pred = [0, predicted_x1[frame], predicted_x2[frame]]
        y_pred = [0, predicted_y1[frame], predicted_y2[frame]]
        line_pred.set_data(x_pred, y_pred)
        time_text_pred.set_text(f'Pred Time: {frame * interval / 1000:.2f}s')

        return line_pred, time_text_pred

    ani = FuncAnimation(fig, update, frames=len(predicted_x2),
                        init_func=init, blit=True, interval=interval)
    return ani

def run_double_pendulum_example():
    print("Starting Double Pendulum Reservoir Computing Example...")

    config_loader = ConfigLoader()

    # 1. Generate Data (Double Pendulum (x,y) movement)
    print("Generating Double Pendulum (x,y) movement data...")
    n_samples = config_loader.get('double_pendulum.timesteps', 5000)
    dt = config_loader.get('double_pendulum.dt', 0.01)
    initial_state = [np.pi/2, 0, np.pi/2, 0] # Initial angles and velocities
    true_x1, true_y1, true_x2, true_y2 = generate_double_pendulum_data(n_samples, dt, initial_state, config_loader=config_loader)
    
    # The data for the RC model will still be just the (x2, y2) coordinates
    data = np.vstack((true_x2, true_y2)).T
    input_dim = data.shape[1] # x, y coordinates
    output_dim = data.shape[1] # x, y coordinates
    print(f"Data generated. Shape: {data.shape}")

    # 2. Initialize Reservoir
    print("Initializing RNN Reservoir...")
    reservoir_dim = config_loader.get('reservoir.n_reservoir', 200)
    spectral_radius = config_loader.get('reservoir.spectral_radius', 0.8)
    sparsity = config_loader.get('reservoir.sparsity', 0.1)
    leaking_rate = config_loader.get('reservoir.leaking_rate', 0.3)
    input_scaling = config_loader.get('reservoir.input_scaling', 0.5)
    random_state = np.random.RandomState(config_loader.get('reservoir.random_state', 42))

    rnn_reservoir = RNNReservoir(
        input_dim=input_dim,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leaking_rate=leaking_rate,
        input_scaling=input_scaling,
        random_state=random_state,
        config_loader=config_loader
    )
    print(f"RNN Reservoir initialized with {reservoir_dim} neurons, spectral radius={spectral_radius}, leaking rate={leaking_rate}, input scaling={input_scaling}.")

    # 3. Initialize Reservoir Computing Model
    print("Initializing Reservoir Computing Model...")
    washout_steps = config_loader.get('trainer.n_drop', 200)
    rc_model = ReservoirComputingModel(
        reservoir=rnn_reservoir,
        output_dim=output_dim,
        washout_steps=washout_steps
    )
    print(f"Model initialized with {washout_steps} washout steps.")

    # 4. Prepare Data for Training and Testing
    print("Preparing data for training and testing...")
    train_ratio = config_loader.get('trainer.train_ratio', 0.7) # Assuming a train_ratio in trainer config
    trainer = Trainer(rc_model)
    X_train, y_train, X_test, y_test = trainer.prepare_data(data, train_ratio=train_ratio, normalize=True)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    # 5. Train and Evaluate
    print("Training and evaluating model...")
    regularization_coeff = config_loader.get('trainer.regularization', 1e-7)
    predictions_train, predictions_test, mse_train, mse_test = trainer.train_and_evaluate(
        X_train, y_train, X_test, y_test, regularization_coeff=regularization_coeff
    )
    print(f"MSE on training data: {mse_train}")
    print(f"MSE on test data: {mse_test}")

    # 6. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    # Plot X-coordinate prediction
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(y_train)), y_train[:, 0], label='True Training X', color='blue')
    plt.plot(np.arange(len(y_train)), predictions_train[:, 0], label='Predicted Training X', color='cyan', linestyle='--')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test[:, 0], label='True Test X', color='green')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test[:, 0], label='Predicted Test X', color='red', linestyle='--')
    plt.title('Double Pendulum X-coordinate Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('X-coordinate')
    plt.legend()
    plt.grid(True)

    # Plot Y-coordinate prediction
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(y_train)), y_train[:, 1], label='True Training Y', color='blue')
    plt.plot(np.arange(len(y_train)), predictions_train[:, 1], label='Predicted Training Y', color='cyan', linestyle='--')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test[:, 1], label='True Test Y', color='green')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test[:, 1], label='Predicted Test Y', color='red', linestyle='--')
    plt.title('Double Pendulum Y-coordinate Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 7. Animate Results
    print("Animating results...")
    # Use the full data for animation, not just test data
    full_true_data_x2 = np.vstack((y_train, y_test))[:, 0]
    full_true_data_y2 = np.vstack((y_train, y_test))[:, 1]
    full_predicted_data_x2 = np.vstack((predictions_train, predictions_test))[:, 0]
    full_predicted_data_y2 = np.vstack((predictions_train, predictions_test))[:, 1]

    # For the predicted pendulum, x1 and y1 are not predicted, so we use the true x1, y1
    # The model only predicts (x2, y2)
    ani_one_step = animate_double_pendulum(true_x1, true_y1, true_x2, true_y2,
                                           true_x1, true_y1, full_predicted_data_x2, full_predicted_data_y2,
                                           interval=dt*1000, config_loader=config_loader)
    plt.show() # Display the one-step-ahead animation

    # 8. Free-running prediction and animation
    print("Generating free-running predictions...")
    # Use a portion of the test data to prime the reservoir
    priming_steps = 200 # Example: use 200 steps from X_test for priming
    free_run_priming_input = X_test[:priming_steps]
    
    # The number of steps to predict in free-running mode
    # Predict for the rest of the test set after priming
    free_run_prediction_steps = len(y_test) - priming_steps 

    # Call free_run_predict
    free_run_predictions = rc_model.free_run_predict(free_run_priming_input, free_run_prediction_steps)

    # Get the corresponding true_x1, true_y1 for the free-run prediction period
    # The free-run predictions start after the priming sequence in the test set.
    # The test set starts at index len(y_train) in the full true_x1, true_y1 arrays.
    start_index_for_free_run_true = len(y_train) + priming_steps
    end_index_for_free_run_true = start_index_for_free_run_true + free_run_prediction_steps

    free_run_true_x1 = true_x1[start_index_for_free_run_true : end_index_for_free_run_true]
    free_run_true_y1 = true_y1[start_index_for_free_run_true : end_index_for_free_run_true]

    # The free_run_predictions are already x2, y2
    free_run_predicted_x2 = free_run_predictions[:, 0]
    free_run_predicted_y2 = free_run_predictions[:, 1]

    print("Animating free-running results...")
    ani_free_run = animate_free_run_pendulum(free_run_true_x1, free_run_true_y1,
                                             free_run_predicted_x2, free_run_predicted_y2,
                                             interval=dt*1000, config_loader=config_loader)
    plt.show() # Display the free-running animation

    print("Double Pendulum example finished.")

if __name__ == "__main__":
    run_double_pendulum_example()
