import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PointFollowing:
    def __init__(self, initial_position=(0.0, 0.0), teleport_interval=100, max_speed=0.1, acceleration_factor=0.01):
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.destination = np.array([0.0, 0.0], dtype=float)
        self.teleport_interval = teleport_interval
        self.max_speed = max_speed
        self.acceleration_factor = acceleration_factor
        self.time_step = 0

    def _teleport_destination(self):
        # Teleport destination to a random point within a reasonable range
        self.destination = np.random.uniform(-5, 5, 2)

    def step(self, model_output):
        # model_output is the desired direction (dx, dy)
        desired_direction = np.array(model_output, dtype=float)
        
        # Normalize desired direction if it's not zero
        if np.linalg.norm(desired_direction) > 1e-6:
            desired_direction = desired_direction / np.linalg.norm(desired_direction)

        # Apply acceleration (optional, based on user request)
        # The model output directly influences the velocity, acting as a force/acceleration
        # Here, we interpret model_output as a desired velocity change or direction.
        # Let's make it influence acceleration.
        
        # Calculate vector towards destination
        vec_to_dest = self.destination - self.position
        
        # Model output is direction coordinates. Let's interpret it as a force/acceleration.
        # The model tries to output a direction that moves the point towards the destination.
        # We can scale this output to act as an acceleration.
        
        # For simplicity, let's assume model_output directly influences velocity for now,
        # and we can add explicit acceleration later if needed.
        # If model_output is direction, let's make it a target velocity direction.
        
        # Let's refine: model_output is 'direction coordinates'.
        # This means the model outputs a vector (vx, vy) that represents the desired velocity.
        # We can then apply this as an acceleration or directly as a velocity.
        
        # Option 1: Model output is desired velocity vector
        # self.velocity = desired_direction * self.max_speed
        
        # Option 2: Model output is an acceleration vector
        # self.velocity += desired_direction * self.acceleration_factor
        # self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        
        # Let's go with Option 2, as it allows for smoother movement and the "acceleration" request.
        # The model output is a direction, so we scale it by acceleration_factor.
        self.velocity += desired_direction * self.acceleration_factor
        
        # Limit velocity to max_speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        self.position += self.velocity

        self.time_step += 1
        if self.time_step % self.teleport_interval == 0:
            self._teleport_destination()

        return self.position, self.destination

def generate_point_following_data(n_samples, teleport_interval=100, max_speed=0.1, acceleration_factor=0.01):
    """
    Generates data for the point following problem.
    Input: destination coordinates
    Output: direction coordinates (to move towards destination)
    """
    point_env = PointFollowing(teleport_interval=teleport_interval, max_speed=max_speed, acceleration_factor=acceleration_factor)
    
    positions = []
    destinations = []
    target_directions = [] # The ideal direction the point should move

    # Initial teleport
    point_env._teleport_destination()

    for _ in range(n_samples):
        positions.append(point_env.position.copy())
        destinations.append(point_env.destination.copy())

        # Calculate the ideal direction: vector from current position to destination
        vec_to_dest = point_env.destination - point_env.position
        
        # Normalize the ideal direction
        if np.linalg.norm(vec_to_dest) > 1e-6:
            ideal_direction = vec_to_dest / np.linalg.norm(vec_to_dest)
        else:
            ideal_direction = np.array([0.0, 0.0]) # If already at destination, no movement

        target_directions.append(ideal_direction)

        # Simulate one step of the environment.
        # For data generation, we'll use the ideal direction as the model_output
        # to see how the point would move if the model was perfect.
        # When training, the model will try to predict this ideal_direction.
        point_env.step(ideal_direction) 
        
    return np.array(positions), np.array(destinations), np.array(target_directions)

def animate_point_following(true_positions, true_destinations, predicted_directions,
                            max_speed=0.1, acceleration_factor=0.01, interval=50):
    """
    Animates the point following behavior.
    true_positions: (N, 2) array of true point positions
    true_destinations: (N, 2) array of true destination positions
    predicted_directions: (N, 2) array of directions predicted by the model
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title('Point Following Simulation: True vs Predicted')
    ax.grid(True)

    # True point and destination
    point_true, = ax.plot([], [], 'o', color='blue', markersize=10, label='True Point')
    destination_true, = ax.plot([], [], 'x', color='green', markersize=12, label='Destination')
    
    # Predicted point (simulated using predicted directions)
    point_pred, = ax.plot([], [], 'o', color='red', markersize=8, label='Predicted Point')
    
    # # Arrow for predicted direction
    # arrow_pred = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc='red', ec='red', alpha=0.7)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend()

    # Initialize a separate PointFollowing environment for the predicted path
    # This environment will use the predicted_directions as its 'model_output'
    predicted_env = PointFollowing(initial_position=true_positions[0], 
                                   teleport_interval=100, # This doesn't matter for predicted path as destination is fixed by true_destinations
                                   max_speed=max_speed, 
                                   acceleration_factor=acceleration_factor)
    predicted_path = [predicted_env.position.copy()]

    def init():
        point_true.set_data([], [])
        destination_true.set_data([], [])
        point_pred.set_data([], [])
        time_text.set_text('')
        # arrow_pred.set_visible(False)
        return point_true, destination_true, point_pred, time_text # Removed arrow_pred from returned artists

    def update(frame):
        # True values
        point_true.set_data([true_positions[frame, 0]], [true_positions[frame, 1]])
        destination_true.set_data([true_destinations[frame, 0]], [true_destinations[frame, 1]])
        
        # Simulate predicted point's movement
        if frame > 0:
            # Update predicted_env's destination to match the true destination for this frame
            predicted_env.destination = true_destinations[frame]
            current_pred_pos, _ = predicted_env.step(predicted_directions[frame-1]) # Use previous frame's prediction for current step
            predicted_path.append(current_pred_pos.copy())
        
        point_pred.set_data([predicted_path[frame][0]], [predicted_path[frame][1]])

        # # Update arrow for predicted direction
        # current_pred_pos = predicted_path[frame]
        # pred_dir = predicted_directions[frame]
        
        # arrow_pred.set_x(current_pred_pos[0])
        # arrow_pred.set_y(current_pred_pos[1])
        # arrow_pred.set_dx(pred_dir[0] * 0.5) # Scale arrow for visibility
        # arrow_pred.set_dy(pred_dir[1] * 0.5) # Scale arrow for visibility
        
        time_text.set_text(f'Time: {frame}')

        return point_true, destination_true, point_pred, time_text

    ani = FuncAnimation(fig, update, frames=len(true_positions),
                        init_func=init, blit=False, interval=interval) # blit=False because arrow is dynamic
    return ani

def run_point_following_example():
    print("Starting Point Following Reservoir Computing Example...")

    # 1. Generate Data
    print("Generating Point Following data...")
    n_samples = 2000
    teleport_interval = 100
    max_speed = 0.1
    acceleration_factor = 0.01 # This is the 'acceleration' requested by the user

    true_positions, true_destinations, target_directions = generate_point_following_data(
        n_samples, teleport_interval, max_speed, acceleration_factor
    )
    
    # Input to the model: current position and destination coordinates
    # Output from the model: direction coordinates (target_directions)
    data_input = np.hstack((true_positions, true_destinations)) # (x_point, y_point, x_dest, y_dest)
    data_output = target_directions # (dx, dy)

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
    
    # For this problem, X and y are distinct.
    # X_train, y_train, X_test, y_test = trainer.prepare_data(data, train_ratio=train_ratio, normalize=True)
    # We need to manually split X (data_input) and y (data_output)
    
    split_idx = int(n_samples * train_ratio)
    
    X_train = data_input[:split_idx]
    y_train = data_output[:split_idx]
    X_test = data_input[split_idx:]
    y_test = data_output[split_idx:]

    # Normalize input data (positions and destinations)
    # We should normalize X_train and X_test based on the min/max of the entire data_input
    min_vals = data_input.min(axis=0)
    max_vals = data_input.max(axis=0)
    
    X_train_normalized = (X_train - min_vals) / (max_vals - min_vals + 1e-8)
    X_test_normalized = (X_test - min_vals) / (max_vals - min_vals + 1e-8)

    # Output data (target_directions) are already normalized vectors, so no further normalization needed for y.
    
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
    full_true_positions = true_positions
    full_true_destinations = true_destinations
    full_predicted_directions = np.vstack((predictions_train, predictions_test))

    animation_obj = animate_point_following(
        full_true_positions, full_true_destinations, full_predicted_directions,
        max_speed=max_speed, acceleration_factor=acceleration_factor, interval=50
    )
    plt.show()

    print("Point Following example finished.")

if __name__ == "__main__":
    run_point_following_example()
