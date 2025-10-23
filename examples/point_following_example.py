import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reservoir_computing.reservoir import RNNReservoir
from reservoir_computing.model import ReservoirComputingModel
from reservoir_computing.trainer import Trainer
from reservoir_computing.point_following import PointFollowing, generate_point_following_data, animate_point_following

def run_point_following_example():
    print("Starting Point Following Reservoir Computing Example...")

    # 1. Generate Data
    print("Generating Point Following data...")
    n_samples = 10000 # Increased number of samples for more training data

    inputs, targets = generate_point_following_data(n_samples)
    
    # Input to the model: current position and destination coordinates
    # Output from the model: direction coordinates (targets)
    data_input = inputs # (x_point, y_point, x_dest, y_dest)
    data_output = targets # (dx, dy)

    input_dim = data_input.shape[1] # 4 (x_point, y_point, x_dest, y_dest)
    output_dim = data_output.shape[1] # 2 (dx, dy)
    print(f"Data generated. Input shape: {data_input.shape}, Output shape: {data_output.shape}")

    # 2. Initialize Reservoir
    print("Initializing RNN Reservoir...")
    reservoir_dim = 300 # Increased reservoir size
    spectral_radius = 0.9
    sparsity = 0.1
    leaking_rate = 0.3
    input_scaling = 0.5
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
    washout_steps = 200
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

    print(f"Training input data shape: {X_train_normalized.shape}, output data shape: {y_train.shape}")
    print(f"Test input data shape: {X_test_normalized.shape}, output data shape: {y_test.shape}")

    # 5. Train and Evaluate
    print("Training and evaluating model...")
    regularization_coeff = 1e-7
    predictions_train, predictions_test, mse_train, mse_test = trainer.train_and_evaluate(
        X_train_normalized, y_train, X_test_normalized, y_test, regularization_coeff=regularization_coeff
    )
    print(f"MSE on training data: {mse_train}")
    print(f"MSE on test data: {mse_test}")

    # 6. Plot Results
    print("Plotting results...")
    fig_coords, axs = plt.subplots(output_dim, 1, figsize=(12, 6), sharex=True)
    coords_labels = ['Direction X', 'Direction Y']

    for i in range(output_dim):
        axs[i].plot(np.arange(len(y_train)), y_train[:, i], label=f'True Training {coords_labels[i]}', color='blue')
        axs[i].plot(np.arange(len(y_train)), predictions_train[:, i], label=f'Predicted Training {coords_labels[i]}', color='cyan', linestyle='--')
        axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test[:, i], label=f'True Test {coords_labels[i]}', color='green')
        axs[i].plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions_test[:, i], label=f'Predicted Test {coords_labels[i]}', color='red', linestyle='--')
        axs[i].set_title(f'Point Following ({coords_labels[i]}) Prediction')
        axs[i].set_ylabel(f'{coords_labels[i]}')
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.show()

    # 7. Animate Results
    print("Animating results...")
    # Only show test results in the animation
    test_inputs = X_test
    test_predicted_directions = predictions_test

    animation_obj = animate_point_following(
        test_inputs, test_predicted_directions, interval=50
    )
    plt.show()

    print("Point Following example finished.")

if __name__ == "__main__":
    run_point_following_example()
