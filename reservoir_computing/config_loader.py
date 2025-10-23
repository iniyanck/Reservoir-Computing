import yaml
import os

class ConfigLoader:
    def __init__(self, config_dir="reservoir_computing/config"):
        self.config_dir = config_dir
        self.config = {}
        self._load_all_configs()

    def _load_all_configs(self):
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.config_dir, filename)
                with open(filepath, 'r') as f:
                    module_config = yaml.safe_load(f)
                    self.config.update(module_config)

    def get(self, key, default=None):
        """
        Retrieves a configuration value using a dot-separated key (e.g., "reservoir.n_reservoir").
        """
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except KeyError:
            return default

    def __getitem__(self, key):
        return self.get(key)

# Example usage:
if __name__ == "__main__":
    loader = ConfigLoader()
    print("Loaded Configuration:")
    print(loader.config)

    # Accessing specific parameters
    print(f"\nReservoir n_reservoir: {loader.get('reservoir.n_reservoir')}")
    print(f"Trainer n_drop: {loader.get('trainer.n_drop')}")
    print(f"Point Following timesteps: {loader.get('point_following.timesteps')}")
    print(f"Double Pendulum g: {loader.get('double_pendulum.g')}")

    # Accessing a non-existent key
    print(f"Non-existent key: {loader.get('non_existent.key', 'default_value')}")
