from reservoir_computing.config_loader import ConfigLoader
from examples.point_following_example import PointFollowing
from examples.double_pendulum_example import DoublePendulum
from examples.rl_training_example import RLTrainingExample
from reservoir_computing.examples.mackey_glass_example import MackeyGlassExample

class ExampleFactory:
    @staticmethod
    def create_example(config_loader: ConfigLoader):
        example_name = config_loader.get('application.name', 'point_following').lower()

        if example_name == 'point_following':
            timesteps = config_loader.get('point_following.timesteps')
            dt = config_loader.get('point_following.dt')
            target_frequency = config_loader.get('point_following.target_frequency')
            amplitude = config_loader.get('point_following.amplitude')
            return PointFollowing(timesteps, dt, target_frequency, amplitude)
        elif example_name == 'double_pendulum':
            timesteps = config_loader.get('double_pendulum.timesteps')
            dt = config_loader.get('double_pendulum.dt')
            g = config_loader.get('double_pendulum.g')
            l1 = config_loader.get('double_pendulum.l1')
            l2 = config_loader.get('double_pendulum.l2')
            return DoublePendulum(config_loader=config_loader)
        elif example_name == 'rl_training':
            # RLTrainingExample might need different initialization or just a placeholder
            return RLTrainingExample(config_loader=config_loader)
        elif example_name == 'mackey_glass':
            return MackeyGlassExample(config_loader=config_loader)
        else:
            raise ValueError(f"Unknown example problem: {example_name}. Supported types: 'point_following', 'double_pendulum', 'rl_training', 'mackey_glass'.")
