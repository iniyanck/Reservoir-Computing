import numpy as np
import matplotlib.pyplot as plt
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.model import ReservoirComputingModel
from reservoir_computing.methods.trainer import TrainerFactory
from reservoir_computing.components.reservoir_factory import ReservoirFactory
from reservoir_computing.examples.example_factory import ExampleFactory
from reservoir_computing.utils import denormalize_data # Import denormalize_data

def main():
    # 1. Load configuration
    config_loader = ConfigLoader()
    
    # 2. Create example problem
    print("Creating example problem...")
    example_problem = ExampleFactory.create_example(config_loader)
    full_data = example_problem.generate_data()
    
    application_name = config_loader.get('application.name', 'point_following').lower()

    if application_name == 'point_following':
        # For point following, full_data is (inputs, targets) stacked
        input_dim = 4 # (x_point, y_point, x_dest, y_dest)
        output_dim = 2 # (dx, dy)
        X = full_data[:, :input_dim]
        y = full_data[:, input_dim:]
    elif application_name == 'double_pendulum':
        # For double pendulum, full_data is (input_data, target_data) stacked
        input_dim = 4 # (x1, y1, x2, y2)
        output_dim = 2 # (x2, y2)
        X = full_data[:, :input_dim]
        y = full_data[:, input_dim:]
    elif application_name == 'rl_training' or application_name == 'mackey_glass':
        # For these, the data is univariate time series, input and output are the same
        input_dim = full_data.shape[1] if full_data.ndim > 1 else 1
        output_dim = input_dim
        X = full_data[:-1]
        y = full_data[1:]
    else:
        raise ValueError(f"Unknown application name: {application_name}")

    # 3. Create reservoir
    print("Creating reservoir...")
    reservoir = ReservoirFactory.create_reservoir(config_loader)
    # Ensure reservoir's input_dim is correctly set
    reservoir.input_dim = input_dim 
    
    # 4. Create model
    print("Creating Reservoir Computing Model...")
    washout_steps = config_loader.get('trainer.n_drop', 0)
    model = ReservoirComputingModel(reservoir, output_dim, washout_steps)

    # 5. Create trainer
    print("Creating Trainer...")
    trainer = TrainerFactory.create_trainer(model, config_loader)

    # 6. Prepare data (split into train/test and normalize)
    print("Preparing data...")
    train_ratio = config_loader.get('global_config.train_ratio', 0.7)
    
    # Trainer's prepare_data expects a single 'data' array and splits it internally
    # For point_following and double_pendulum, we already split X and y.
    # We need to adapt trainer.prepare_data or call it differently.
    # Let's modify trainer.prepare_data to accept X and y directly.
    # For now, I'll manually split here and pass to train_and_evaluate/train_rl_continually
    
    num_timesteps = X.shape[0]
    num_train = int(num_timesteps * train_ratio)

    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]

    # Normalize data using the trainer's utility
    # The trainer's prepare_data handles normalization and min/max tracking.
    # We need to call it with the combined data if we want it to handle normalization.
    # Or, we normalize X and y separately.
    # Let's stick to the trainer's prepare_data for consistency, but pass the combined data.
    
    # Re-combine X and y for trainer.prepare_data if it expects a single array
    # This is a bit redundant, but keeps trainer.prepare_data simple.
    # The trainer's prepare_data assumes y is the next timestep of X.
    # This is not true for point_following/double_pendulum where y is a target.
    # So, we need a different approach for prepare_data or handle normalization here.

    # Let's modify trainer.prepare_data to accept X and y directly, and normalize them.
    # For now, I will manually normalize X and y here.
    
    # Manual Normalization (if trainer.prepare_data is not used for this)
    original_min_X, original_max_X = np.min(X), np.max(X)
    original_min_y, original_max_y = np.min(y), np.max(y)
    
    X_normalized = (X - original_min_X) / (original_max_X - original_min_X + 1e-8)
    y_normalized = (y - original_min_y) / (original_max_y - original_min_y + 1e-8)

    X_train_norm, y_train_norm = X_normalized[:num_train], y_normalized[:num_train]
    X_test_norm, y_test_norm = X_normalized[num_train:], y_normalized[num_train:]

    # Store original min/max in trainer for denormalization
    trainer.original_min_X = original_min_X
    trainer.original_max_X = original_max_X
    trainer.original_min_y = original_min_y
    trainer.original_max_y = original_max_y

    # 7. Run training
    print("Running training...")
    predictions_train, predictions_test, mse_train, mse_test = trainer.run_training(X_train_norm, y_train_norm, X_test_norm, y_test_norm)

    print("\n--- Training Results ---")
    print(f"Training Method: {config_loader.get('trainer.method')}")
    print(f"Reservoir Type: {config_loader.get('reservoir.type')}")
    print(f"Example Problem: {config_loader.get('application.name')}")
    print(f"Mean Squared Error (Train): {mse_train:.4f}")
    print(f"Mean Squared Error (Test): {mse_test:.4f}")

    # 8. Plot results (optional)
    plt.figure(figsize=(12, 6))
    
    # Denormalize y_test for plotting
    y_test_denorm = denormalize_data(y_test_norm, trainer.original_min_y, trainer.original_max_y)

    if output_dim == 1: # Mackey-Glass, RL Training
        plt.plot(example_problem.time_steps[num_train+1:], y_test_denorm[:, 0], label='Actual Test Data')
        plt.plot(example_problem.time_steps[num_train+1:], predictions_test[:, 0], label='Predicted Test Data')
        plt.title(f"Reservoir Computing Prediction for {application_name} (Test Set)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
    elif output_dim == 2: # Point Following, Double Pendulum
        fig_coords, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        labels = ['Dim 1', 'Dim 2']
        if application_name == 'point_following':
            labels = ['Direction X', 'Direction Y']
        elif application_name == 'double_pendulum':
            labels = ['X2 Coordinate', 'Y2 Coordinate']

        for i in range(output_dim):
            axs[i].plot(example_problem.time_steps[num_train:], y_test_denorm[:, i], label=f'True Test {labels[i]}', color='blue')
            axs[i].plot(example_problem.time_steps[num_train:], predictions_test[:, i], label=f'Predicted Test {labels[i]}', color='red', linestyle='--')
            axs[i].set_title(f"{application_name} {labels[i]} Prediction (Test Set)")
            axs[i].set_ylabel(labels[i])
            axs[i].legend()
            axs[i].grid(True)
        axs[-1].set_xlabel("Time")
        plt.tight_layout()
    else:
        print("Plotting not implemented for this output dimension.")

    plt.show()

    # Handle animations
    if application_name == 'point_following':
        from examples.point_following_example import animate_point_following
        
        predictions_full_denorm = denormalize_data(np.vstack((predictions_train, predictions_test)), trainer.original_min_y, trainer.original_max_y)
        inputs_for_animation = X # Original X, not normalized
        
        print("Animating Point Following results...")
        ani = animate_point_following(inputs_for_animation, predictions_full_denorm, config_loader=config_loader)
        plt.show()
    elif application_name == 'double_pendulum':
        from examples.double_pendulum_example import animate_double_pendulum, animate_free_run_pendulum
        
        predictions_full_denorm = denormalize_data(np.vstack((predictions_train, predictions_test)), trainer.original_min_y, trainer.original_max_y)
        
        dp_instance = example_problem
        _, true_x1_full, true_y1_full, true_x2_full, true_y2_full = dp_instance.simulate(dp_instance.initial_state, dp_instance.time_steps)

        predicted_x1_full = true_x1_full
        predicted_y1_full = true_y1_full
        predicted_x2_full = predictions_full_denorm[:, 0]
        predicted_y2_full = predictions_full_denorm[:, 1]

        print("Animating Double Pendulum one-step-ahead results...")
        ani_one_step = animate_double_pendulum(true_x1_full, true_y1_full, true_x2_full, true_y2_full,
                                               predicted_x1_full, predicted_y1_full, predicted_x2_full, predicted_y2_full,
                                               L1=dp_instance.L1, L2=dp_instance.L2, interval=dp_instance.dt*1000)
        plt.show()

        print("Generating free-running predictions...")
        priming_steps = config_loader.get('double_pendulum.priming_steps', 200)
        free_run_prediction_steps = len(y_test) - priming_steps
        
        free_run_priming_input = X_test_norm[:priming_steps]
        free_run_true_input_for_prediction = X_test_norm[priming_steps : priming_steps + free_run_prediction_steps, :input_dim] # Get the true input for the prediction phase

        free_run_predictions_norm = model.free_run_predict(free_run_priming_input, free_run_prediction_steps, free_run_true_input_for_prediction)
        free_run_predictions_denorm = denormalize_data(free_run_predictions_norm, trainer.original_min_y, trainer.original_max_y)

        start_index_for_free_run_true = num_train + priming_steps
        end_index_for_free_run_true = start_index_for_free_run_true + free_run_prediction_steps

        free_run_true_x1 = true_x1_full[start_index_for_free_run_true : end_index_for_free_run_true]
        free_run_true_y1 = true_y1_full[start_index_for_free_run_true : end_index_for_free_run_true]

        free_run_predicted_x2 = free_run_predictions_denorm[:, 0]
        free_run_predicted_y2 = free_run_predictions_denorm[:, 1]

        print("Animating free-running results...")
        ani_free_run = animate_free_run_pendulum(free_run_true_x1, free_run_true_y1,
                                                 free_run_predicted_x2, free_run_predicted_y2,
                                                 L1=dp_instance.L1, L2=dp_instance.L2, interval=dp_instance.dt*1000)
        plt.show()

if __name__ == "__main__":
    main()
