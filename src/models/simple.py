"""
Python module that defines Simple LSTM Based NN module.
"""
import torch
import torch.nn as nn

from src import definitions
from src.bin import validations


class SimpleLSTM(nn.Module):
    """
    Simple LSTM followed by a dense layer for predicting time series.
    """

    def __init__(self, num_features,
                 num_classes=3,
                 hidden_size=64,
                 dropout=0,
                 bidirectional=False,
                 covariates=0):
        super(SimpleLSTM, self).__init__()
        self.covariates = covariates
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        dense_layer_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        dense_layer_hidden_size = dense_layer_hidden_size + covariates if covariates > 0 else dense_layer_hidden_size
        self.linear = nn.Linear(dense_layer_hidden_size, num_classes)

    def forward(self, tensor_data, covariates=None):
        assert covariates is not None and self.covariates > 0 or covariates is None and self.covariates == 0,\
            "If training for covariates, initialize correctly."

        if covariates is not None:
            assert covariates.shape[0] == self.covariates, "Expected covariate size and input mismatch."
        # Extracting actual data form the tuple.
        input_sequence = tensor_data[definitions.ACTUAL_DATA_IDX].unsqueeze(0)
        validations.validate_no_nans_in_tensor(input_sequence)

        lstm_out, hidden = self.lstm(input_sequence)
        y_out = lstm_out[:, -1].unsqueeze(0)

        if self.covariates > 0 and covariates is not None:
            # Adding two dummy dimensions.
            covariates = covariates.unsqueeze(0).unsqueeze(0)
            y_out = torch.cat((y_out, covariates), dim=2)

        y_out = self.linear(y_out)
        y_out = y_out.squeeze(0).squeeze(0)

        return y_out


class SimpleCNN(nn.Module):
    """
    Simple LSTM followe by a dense layer for predicting time series.
    """

    def __init__(self,
                 num_features=14,
                 sequence_length=72,
                 num_classes=3,
                 in_channels=3,
                 out_channels=24,
                 # This kernel configuration allows us to take. (_, num_features)
                 kernel_size=(12, 14),
                 stride=1,
                 padding=0,
                 bias=True):
        super(SimpleCNN, self).__init__()
        assert kernel_size[1] == num_features, \
            "Num Features and Kernel Size mismatch. Expected kernel size (_ , {}) for {}".format(num_features,
                                                                                                 kernel_size)
        kernel_height, kernel_width = kernel_size
        expected_h = ((sequence_length - kernel_height + 2 * padding) / stride) + 1
        expected_w = ((num_features - kernel_width + 2 * padding) / stride) + 1
        assert int(expected_h) == expected_h and int(expected_w) == expected_w, "CNN parameters o not fit sequence."

        self.cnn2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.linear = nn.Linear(int(out_channels * expected_h * expected_w), num_classes)
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.num_features = num_features

    def forward(self, tensor_data, covariate=None):
        tensor_data = tensor_data.unsqueeze(0)
        validations.validate_no_nans_in_tensor(tensor_data)
        assert tensor_data.shape[1] >= self.in_channels and tensor_data.shape[2] == self.sequence_length and tensor_data.shape[3] == self.num_features, "Wrong dimensions in input data!"

        y_out = self.cnn2d(tensor_data)
        y_out = y_out.reshape(-1)
        y_out = self.linear(y_out)
        y_out = nn.functional.relu(y_out)

        return y_out
