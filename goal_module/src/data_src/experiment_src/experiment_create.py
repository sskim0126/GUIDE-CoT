from src.data_src.experiment_src.experiment_eth5 import Experiment_eth5


def create_experiment(dataset_name):
    if dataset_name.lower() == 'eth5':
        return Experiment_eth5
    else:
        raise NotImplementedError("Experiment object not available yet!")
