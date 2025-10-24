import torch
from reservoir_computing.components.reservoirs.rnn_reservoir import RNNReservoir
from reservoir_computing.config_loader import ConfigLoader

class ReservoirFactory:
    @staticmethod
    def create_reservoir(config_loader: ConfigLoader, device=None):
        reservoir_type = config_loader.get('reservoir.type', 'rnn').lower()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if reservoir_type == 'rnn':
            # RNNReservoir already loads its parameters from config_loader internally
            return RNNReservoir(config_loader=config_loader, device=device)
        # Add other reservoir types here as they are implemented
        # elif reservoir_type == 'snn':
        #     return SNNReservoir(config_loader=config_loader)
        else:
            raise ValueError(f"Unknown reservoir type: {reservoir_type}. Supported types: 'rnn'.")
