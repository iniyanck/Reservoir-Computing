import yaml
import os

class ConfigLoader:
    def __init__(self):
        self.config = {}
        self._load_global_config()
        self._load_all_configs()

    def _load_global_config(self):
        global_config_path = os.path.join("reservoir_computing/config", "global_config.yaml")
        with open(global_config_path, 'r') as f:
            global_config = yaml.safe_load(f)
            self.config.update(global_config)
        self.config_dir = self.config['paths']['config_dir']

    def _load_all_configs(self):
        # Ensure global_config.yaml is not loaded again
        for filename in os.listdir(self.config_dir):
            if (filename.endswith(".yaml") or filename.endswith(".yml")) and filename != "global_config.yaml":
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

    # Accessing paths from global config
    print(f"\nConfig directory: {loader.get('paths.config_dir')}")
    print(f"Methods directory: {loader.get('paths.methods_dir')}")
    print(f"Components directory: {loader.get('paths.components_dir')}")
    print(f"Examples directory: {loader.get('paths.examples_dir')}")

    # Accessing specific parameters
    print(f"\nReservoir n_reservoir: {loader.get('reservoir.n_reservoir')}")
    print(f"Trainer n_drop: {loader.get('trainer.n_drop')}")
    print(f"Point Following timesteps: {loader.get('point_following.timesteps')}")
    print(f"Double Pendulum g: {loader.get('double_pendulum.g')}")

    # Accessing a non-existent key
    print(f"Non-existent key: {loader.get('non_existent.key', 'default_value')}")
