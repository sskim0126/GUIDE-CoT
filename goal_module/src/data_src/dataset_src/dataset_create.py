from src.data_src.dataset_src.dataset_eth5 import Dataset_eth5


def create_dataset(dataset_name):
    if dataset_name.lower() == 'eth5':
        return Dataset_eth5()
    else:
        raise NotImplementedError("Dataset object not available yet!")
