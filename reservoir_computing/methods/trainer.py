from reservoir_computing.config_loader import ConfigLoader
from reservoir_computing.components.model import ReservoirComputingModel
from reservoir_computing.methods.trainers.regression_trainer import RegressionTrainer
from reservoir_computing.methods.trainers.rl_trainer import RLTrainer
from reservoir_computing.methods.trainers.base_trainer import BaseTrainer

class TrainerFactory:
    @staticmethod
    def create_trainer(model: ReservoirComputingModel, config_loader: ConfigLoader = None) -> BaseTrainer:
        training_method = config_loader.get('trainer.method', 'regression').lower()
        
        if training_method == 'regression':
            return RegressionTrainer(model, config_loader)
        elif training_method == 'rl':
            return RLTrainer(model, config_loader)
        else:
            raise ValueError(f"Unknown training method: {training_method}. Supported types: 'regression', 'rl'.")
