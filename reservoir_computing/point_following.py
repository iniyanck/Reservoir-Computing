import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reservoir_computing.reservoir import RNNReservoir
from .config_loader import ConfigLoader

class PointFollowing:
    def __init__(self, initial_position=(0.0, 0.0), config_loader=None):
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.destination = np.array([0.0, 0.0], dtype=float)
        
        self.teleport_interval = self.config_loader.get('point_following.teleport_interval', 100)
        self.max_speed = self.config_loader.get('point_following.max_speed', 0.1)
        self.acceleration_factor = self.config_loader.get('point_following.acceleration_factor', 0.01)
        
        self.time_step = 0
        # Generate four random destination points once during initialization
        # These points will be constant throughout the training run
        self.destination_points = [np.random.uniform(-3, 3, 2) for _ in range(4)]

    def _teleport_destination(self):
        # Randomly choose one of the four pre-generated destination points
        self.destination = self.destination_points[np.random.randint(len(self.destination_points))]
        # Do not reset position and velocity to origin, let the point continue from where it left off.
        # self.position = np.array([0.0, 0.0], dtype=float)
        # self.velocity = np.array([0.0, 0.0], dtype=float)

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

def generate_point_following_data(n_samples, seed=42, config_loader=None):
    np.random.seed(seed)
    
    if config_loader is None:
        config_loader = ConfigLoader()

    point_follower = PointFollowing(config_loader=config_loader)
    
    inputs = [] # (current_position, destination)
    targets = [] # direction of movement

    for _ in range(n_samples):
        # Input to the model will be the current position and the destination
        current_input = np.concatenate((point_follower.position, point_follower.destination))
        inputs.append(current_input)

        # Calculate the ideal direction towards the destination (this will be the target output)
        vec_to_dest = point_follower.destination - point_follower.position
        if np.linalg.norm(vec_to_dest) > 1e-6:
            ideal_direction = vec_to_dest / np.linalg.norm(vec_to_dest)
        else:
            ideal_direction = np.array([0.0, 0.0]) # No movement if at destination

        targets.append(ideal_direction)
        
        # Simulate one step of the point follower using the ideal direction
        # This advances the point_follower's state for the next iteration
        point_follower.step(ideal_direction) 

    return np.array(inputs), np.array(targets)

def animate_point_following(inputs, predicted_directions, interval=50, config_loader=None):
    if config_loader is None:
        config_loader = ConfigLoader()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Point Following Animation (Predicted Path)')
    ax.grid(True)

    predicted_point, = ax.plot([], [], 'x', color='red', label='Predicted Point')
    destination_marker, = ax.plot([], [], 'D', color='green', markersize=10, label='Destination')
    
    ax.legend()

    # Initialize a PointFollowing instance for the predicted path
    predicted_follower = PointFollowing(config_loader=config_loader)
    # Set initial position and destination from the first input
    predicted_follower.position = inputs[0, :2]
    predicted_follower.destination = inputs[0, 2:]
    # Ensure its initial destination points match the sequence from inputs for consistency
    # This is a bit tricky as the destination points are generated internally by PointFollowing.
    # For animation, we will directly set the destination from the input data.

    def init():
        predicted_point.set_data([], [])
        destination_marker.set_data([], [])
        return predicted_point, destination_marker

    def update(frame):
        # Get current position and destination from the input data for this frame
        current_position_input = inputs[frame, :2]
        current_destination_input = inputs[frame, 2:]

        # Update the predicted_follower's state based on the input for this frame
        # We need to ensure the predicted_follower's internal state (position, destination)
        # is consistent with the input data at the start of each frame's calculation.
        # However, the predicted_follower's position is updated by its own step method.
        # The destination should be taken from the input for consistency.
        predicted_follower.destination = current_destination_input
        
        # The teleport logic is now handled by the PointFollowing instance itself,
        # and the point should continue from where it left off.
        # We only need to ensure the destination is updated.

        # Update predicted point using the predicted directions from the model
        predicted_follower.step(predicted_directions[frame])
        predicted_point.set_data([predicted_follower.position[0]], [predicted_follower.position[1]])
        
        # Update destination marker
        destination_marker.set_data([current_destination_input[0]], [current_destination_input[1]])

        return predicted_point, destination_marker

    ani = FuncAnimation(fig, update, frames=len(inputs),
                        init_func=init, blit=True, interval=interval, repeat=False)
    return ani # Return the animation object to prevent it from being garbage collected
