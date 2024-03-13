import torch

from src import definitions
from src.bin import validations


def get_data_and_label_tensor(data: dict, key, cuda_enabled):
    """

    @param data: Data dict containing the data in our rich data structure.
    @param key: Key in the data, usually time series key.
    @param  cuda_enabled: If true, returns cuda tensors.
    @return: Returns tensors that can be used for training on the models.
    """
    tensor_data = torch.tensor(list(data['data'][key][:definitions.COVARIATE_DATA_IDX]),
                               dtype=torch.float)
    covariate_data = torch.tensor(list(data['data'][key][definitions.COVARIATE_DATA_IDX]),
                                  dtype=torch.float)
    histogram_data = torch.tensor(list(data['data'][key][definitions.HISTOGRAM_IDX]),
                                  dtype=torch.float)
    train_label = torch.tensor(data['data'][key][definitions.LABELS_IDX]).item()
    train_label = torch.tensor([train_label], dtype=torch.long)

    if cuda_enabled:
        tensor_data = tensor_data.cuda()
        covariate_data = covariate_data.cuda()
        histogram_data = histogram_data.cuda()
        train_label = train_label.cuda()

    return tensor_data, covariate_data, histogram_data, train_label


def tensorify_data_gru_d(data: dict, cuda_enabled=False):
    """

    @param data: Data dictionary that needs to be converted to tensors in GRUD style of data.
    @param cuda_enabled: If true, will convert data into cuda tensors.
    @return: Return Data dictionary with tensors which can be used to train.
    """
    validations.validate_data_dict_keys(data)
    validations.validate_all_data_present_in_data_dict(data)
    for key in data['data'].keys():
        data['data'][key] = get_data_and_label_tensor(data, key, cuda_enabled)

    return data
