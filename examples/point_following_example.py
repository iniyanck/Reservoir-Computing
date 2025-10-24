import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.example import Example # Import the abstract base class

class PointFollowing(Example): # Inherit from Example
    def __init__(self, config_loader: ConfigLoader = None):
        super().__init__() # Call the base class constructor
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader

        self.initial_position = np.array([0.0, 0.0], dtype=float)
        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.destination = np.array([0.0, 0.0], dtype=float)
        
        self.timesteps = self.config_loader.get('point_following.timesteps', 1000)
        self.teleport_interval = self.config_loader.get('point_following.teleport_interval', 100)
        self.max_speed = self.config_loader.get('point_following.max_speed', 0.1)
        self.acceleration_factor = self.config_loader.get('point_following.acceleration_factor', 0.01)
        
        self.time_step_counter = 0 # Renamed to avoid conflict with self.time_steps array
        self.time_steps = np.arange(self.timesteps) # For plotting and data generation

        # Generate four random destination points once during initialization
        self.destination_points = [np.random.uniform(-3, 3, 2) for _ in range(4)]
        self._teleport_destination() # Set initial destination

    def _teleport_destination(self):
        self.destination = self.destination_points[np.random.randint(len(self.destination_points))]

    def step(self, model_output):
        desired_direction = np.array(model_output, dtype=float)
        
        if np.linalg.norm(desired_direction) > 1e-6:
            desired_direction = desired_direction / np.linalg.norm(desired_direction)

        self.velocity += desired_direction * self.acceleration_factor
        
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        self.position += self.velocity

        self.time_step_counter += 1
        if self.time_step_counter % self.teleport_interval == 0:
            self._teleport_destination()

        return self.position, self.destination

    def generate_data(self):
        """
        Generates the input and target data for the point following problem.
        Returns:
            np.ndarray: Stacked input and target data.
                        Input: (current_position, destination)
                        Target: ideal_direction
        """
        np.random.seed(42) # Ensure reproducibility for data generation

        # Reset point follower state for data generation
        self.position = self.initial_position.copy()
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.time_step_counter = 0
        self._teleport_destination() # Reset destination

        inputs = [] # (current_position, destination)
        targets = [] # direction of movement

        for _ in range(self.timesteps):
            current_input = np.concatenate((self.position, self.destination))
            inputs.append(current_input)

            vec_to_dest = self.destination - self.position
            if np.linalg.norm(vec_to_dest) > 1e-6:
                ideal_direction = vec_to_dest / np.linalg.norm(vec_to_dest)
            else:
                ideal_direction = np.array([0.0, 0.0])

            targets.append(ideal_direction)
            
            self.step(ideal_direction) # Simulate one step using the ideal direction

        # The main.py expects a single data array. We'll stack inputs and targets.
        # The trainer will then split them.
        # For point following, the input to the model is (position, destination) and output is (direction).
        # So, we return a combined array where the first 4 columns are input and last 2 are target.
        return np.hstack((np.array(inputs), np.array(targets)))


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

    predicted_follower = PointFollowing(config_loader=config_loader)
    predicted_follower.position = inputs[0, :2]
    predicted_follower.destination = inputs[0, 2:]
    predicted_follower.time_step_counter = 0 # Reset counter for animation

    def init():
        predicted_point.set_data([], [])
        destination_marker.set_data([], [])
        return predicted_point, destination_marker

    def update(frame):
        current_position_input = inputs[frame, :2]
        current_destination_input = inputs[frame, 2:]

        predicted_follower.destination = current_destination_input
        
        predicted_follower.step(predicted_directions[frame])
        predicted_point.set_data([predicted_follower.position[0]], [predicted_follower.position[1]])
        
        destination_marker.set_data([current_destination_input[0]], [current_destination_input[1]])

        return predicted_point, destination_marker

    ani = FuncAnimation(fig, update, frames=len(inputs),
                        init_func=init, blit=True, interval=interval, repeat=False)
    return ani
