import numpy as np
import matplotlib.pyplot as plt
from reservoir_computing.reservoir import RNNReservoir
from reservoir_computing.model import ReservoirComputingModel
from reservoir_computing.trainer import Trainer
from reservoir_computing.double_pendulum import generate_double_pendulum_data, animate_double_pendulum
from reservoir_computing.point_following import generate_point_following_data, animate_point_following, PointFollowing

def generate_mackey_glass(n_samples, tau=17, delay=100, seed=42):
    """
    Generates a Mackey-Glass time series.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples + delay)
    x[0:delay] = 0.5 + 0.1 * np.random.rand(delay) # Initial conditions
    
    a = 0.2
    b = 0.1
    gamma = 1.0
    n = 10

    for i in range(delay, n_samples + delay -1):
        x_tau = x[i - tau]
        x[i] = x[i-1] + (a * x_tau / (1 + x_tau**n) - b * x[i-1]) / gamma
    return x[delay:].reshape(-1, 1)

def run_example(problem_type="mackey_glass"):
    print(f"Starting Reservoir Computing Example for {problem_type}...")

    if problem_type == "mackey_glass":
        # 1. Generate Data (Mackey-Glass time series)
        print("Generating Mackey-Glass time series data...")
        n_samples = 2000
        data = generate_mackey_glass(n_samples)
        input_dim = data.shape[1]
        output_dim = data.shape[1]
        reservoir_dim = 100
        spectral_radius = 0.9
        leaking_rate = 0.3
        input_scaling = 0.5
        washout_steps = 100
        regularization_coeff = 1e-8
        title = 'Mackey-Glass Time Series Prediction using Reservoir Computing'
        y_label = 'Value'
    elif problem_type == "double_pendulum":
        # 1. Generate Data (Double Pendulum (x,y) movement)
        print("Generating Double Pendulum (x,y) movement data...")
        n_samples = 5000
        dt = 0.02
        initial_state = [np.pi/2, 0, np.pi/2, 0] # Initial angles and velocities
        true_x1, true_y1, true_x2, true_y2 = generate_double_pendulum_data(n_samples, dt, initial_state)
        
        # The data for the RC model will now predict (x1, y1, x2, y2)
        data = np.vstack((true_x1, true_y1, true_x2, true_y2)).T
        input_dim = data.shape[1] # x1, y1, x2, y2 coordinates
        output_dim = data.shape[1] # x1, y1, x2, y2 coordinates
        reservoir_dim = 200 # Increased reservoir size for more complex dynamics
        spectral_radius = 0.8
        leaking_rate = 0.3
        input_scaling = 0.5
        washout_steps = 200 # Increased washout steps
        regularization_coeff = 1e-7 # Adjusted regularization
        title = 'Double Pendulum (X1,Y1,X2,Y2) Prediction using Reservoir Computing'
        y_label = 'Coordinate Value'
    elif problem_type == "point_following":
        # 1. Generate Data (Point Following)
        print("Generating Point Following data...")
        n_samples = 2000
        teleport_interval = 100
        max_speed = 0.1
        acceleration_factor = 0.01

        true_positions, true_destinations, target_directions = generate_point_following_data(
            n_samples, teleport_interval, max_speed, acceleration_factor
        )
        
        # Input to the model: current position and destination coordinates
        # Output from the model: direction coordinates (target_directions)
        data_input = np.hstack((true_positions, true_destinations)) # (x_point, y_point, x_dest, y_dest)
        data_output = target_directions # (dx, dy)

        input_dim = data_input.shape[1] # 4 (x_point, y_point, x_dest, y_dest)
        output_dim = data_output.shape[1] # 2 (dx, dy)
        reservoir_dim = 300 # Increased reservoir size
        spectral_radius = 0.9
        leaking_rate = 0.3
        input_scaling = 0.5
        washout_steps = 200
        regularization_coeff = 1e-7
        title = 'Point Following Direction Prediction using Reservoir Computing'
        y_label = 'Direction Value'
    else:
        raise ValueError("Invalid problem_type. Choose 'mackey_glass', 'double_pendulum', or 'point_following'.")

    print(f"Data generated. Input shape: {data_input.shape}, Output shape: {data_output.shape}")

    # 2. Initialize Reservoir
    print("Initializing RNN Reservoir...")
    sparsity = 0.1        # Fraction of recurrent connections that are zero
    random_state = np.random.RandomState(42)

    rnn_reservoir = RNNReservoir(
        input_dim=input_dim,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leaking_rate=leaking_rate,
        input_scaling=input_scaling,
        random_state=random_state
    )
    print(f"RNN Reservoir initialized with {reservoir_dim} neurons, spectral radius={spectral_radius}, leaking rate={leaking_rate}, input scaling={input_scaling}.")

    # 3. Initialize Reservoir Computing Model
    print("Initializing Reservoir Computing Model...")
    rc_model = ReservoirComputingModel(
        reservoir=rnn_reservoir,
        output_dim=output_dim,
        washout_steps=washout_steps
    )
    print(f"Model initialized with {washout_steps} washout steps.")

    # 4. Prepare Data for Training and Testing
    print("Preparing data for training and testing...")
    train_ratio = 0.7
    trainer = Trainer(rc_model)
    
    if problem_type == "point_following":
        split_idx = int(n_samples * train_ratio)
        
        X_train = data_input[:split_idx]
        y_train = data_output[:split_idx]
        X_test = data_input[split_idx:]
        y_test = data_output[split_idx:]

        # Normalize input data (positions and destinations)
        min_vals = data_input.min(axis=0)
        max_vals = data_input.max(axis=0)
        
        X_train_normalized = (X_train - min_vals) / (max_vals - min_vals + 1e-8)
        X_test_normalized = (X_test - min_vals) / (max_vals - min_vals + 1e-8)
        
        X_train = X_train_normalized
        X_test = X_test_normalized
        # y_train and y_test (target_directions) are already normalized vectors.
    else:
        X_train, y_train, X_test, y_test = trainer.prepare_data(data, train_ratio=train_ratio, normalize=True)
    
    print(f"Training input data shape: {X_train.shape}, output data shape: {y_train.shape}")
    print(f"Test input data shape: {X_test.shape}, output data shape: {y_test.shape}")

    # 5. Train and Evaluate
    print("Training and evaluating model...")
    predictions_train, predictions_test, mse_train, mse_test = trainer.train_and_evaluate(
        X_train, y_train, X_test, y_test, regularization_coeff=regularization_coeff
    )
    print(f"MSE on training data: {mse_train}")
    print(f"MSE on test data: {mse_test}")

    # 6. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(12, 6))

    if output_dim == 1: # Mackey-Glass
        plt.plot(np.arange(len(y_train)), y_train, label='True Training Data', color='blue')
        plt.plot(np.arange(len(y_train)), predictions_train, label='Predicted Training Data', color='cyan', linestyle='--')
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='True Test Data', color='green')
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test, label='Predicted Test Data', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
    elif output_dim == 4: # Double Pendulum (x1,y1,x2,y2)
        fig_coords, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        coords_labels = ['X1', 'Y1', 'X2', 'Y2']

        for i in range(output_dim):
            axs[i].plot(np.arange(len(y_train)), y_train[:, i], label=f'True Training {coords_labels[i]}', color='blue')
            axs[i].plot(np.arange(len(y_train)), predictions_train[:, i], label=f'Predicted Training {coords_labels[i]}', color='cyan', linestyle='--')
            axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test[:, i], label=f'True Test {coords_labels[i]}', color='green')
            axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test[:, i], label=f'Predicted Test {coords_labels[i]}', color='red', linestyle='--')
            axs[i].set_title(f'{title} ({coords_labels[i]}-coordinate)')
            axs[i].set_ylabel(f'{coords_labels[i]}-coordinate')
            axs[i].legend()
            axs[i].grid(True)
        
        axs[-1].set_xlabel('Time Steps')
        plt.tight_layout()
    elif problem_type == "point_following": # Point Following (Direction X, Direction Y)
        fig_coords, axs = plt.subplots(output_dim, 1, figsize=(12, 6), sharex=True)
        coords_labels = ['Direction X', 'Direction Y']

        for i in range(output_dim):
            axs[i].plot(np.arange(len(y_train)), y_train[:, i], label=f'True Training {coords_labels[i]}', color='blue')
            axs[i].plot(np.arange(len(y_train)), predictions_train[:, i], label=f'Predicted Training {coords_labels[i]}', color='cyan', linestyle='--')
            axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test[:, i], label=f'True Test {coords_labels[i]}', color='green')
            axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test[:, i], label=f'Predicted Test {coords_labels[i]}', color='red', linestyle='--')
            axs[i].set_title(f'{title} ({coords_labels[i]}) Prediction')
            axs[i].set_ylabel(f'{coords_labels[i]}')
            axs[i].legend()
            axs[i].grid(True)
        
        axs[-1].set_xlabel('Time Steps')
        plt.tight_layout()
    
    # If there's an animation, store it to prevent it from being garbage collected
    # and ensure it displays when plt.show() is called.
    animation_obj = None
    if problem_type == "double_pendulum":
        print("Animating results...")
        # Use the full data for animation, not just test data
        full_true_data = np.vstack((y_train, y_test))
        full_predicted_data = np.vstack((predictions_train, predictions_test))
        
        # Extract true coordinates (already available from generate_double_pendulum_data)
        # Note: true_x1, true_y1, true_x2, true_y2 are already defined in this scope
        # from the initial call to generate_double_pendulum_data
        
        # Extract predicted coordinates from the model's output
        predicted_x1_full = full_predicted_data[:, 0]
        predicted_y1_full = full_predicted_data[:, 1]
        predicted_x2_full = full_predicted_data[:, 2]
        predicted_y2_full = full_predicted_data[:, 3]
        
        animation_obj = animate_double_pendulum(
            true_x1, true_y1, true_x2, true_y2,
            predicted_x1_full, predicted_y1_full, predicted_x2_full, predicted_y2_full,
            L1=1.0, L2=1.0, interval=dt*1000
        )
    elif problem_type == "point_following":
        print("Animating results...")
        # The true_positions, true_destinations, and target_directions are available from data generation
        # We need to pass the full predicted directions
        full_predicted_directions = np.vstack((predictions_train, predictions_test))
        
        animation_obj = animate_point_following(
            true_positions, true_destinations, full_predicted_directions,
            max_speed=max_speed, acceleration_factor=acceleration_factor, interval=50
        )

    plt.show() # This will display all figures and animations
    print("Example finished.")

if __name__ == "__main__":
    # To run Mackey-Glass example:
    # run_example(problem_type="mackey_glass")
    # To run Double Pendulum example:
    # run_example(problem_type="double_pendulum")
    # To run Point Following example:
    run_example(problem_type="point_following")
