import torch
import torch.nn as nn

from src.models import autoencoder
from src.bin import validations


class AutoEncoderClassifier(nn.Module):
    def __init__(self,
                 users: list,
                 autoencoder_input_size,
                 autoencoder_bottleneck_feature_size,
                 autoencoder_num_layers,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 num_covariates=0,
                 shared_layer_dropout_prob=0,
                 user_head_dropout_prob=0,
                 ordinal_regression_head=False,
                 bidirectional=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param autoencoder_input_size: Input size of the time series portion on the model.
        @param autoencoder_bottleneck_feature_size: Encoded input size of autoecoder.
        @param autoencoder_num_layers: Num layers in autoencoder LSTM model.
        @param user_dense_layer_hidden_size: dense head hidden size. Same for all users.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(AutoEncoderClassifier, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.autoencoder_input_size = autoencoder_input_size
        self.autoencoder_bottleneck_feature_size = autoencoder_bottleneck_feature_size
        self.autoencoder_num_layers = autoencoder_num_layers
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates
        self.shared_layer_dropout_prob = shared_layer_dropout_prob
        self.user_head_dropout_prob = user_head_dropout_prob
        self.ordinal_regression_head = ordinal_regression_head
        self.bidirectional = bidirectional

        # Layer initialization.
        self.autoencoder = autoencoder.LSTMAE(self.autoencoder_input_size,
                                              self.autoencoder_bottleneck_feature_size,
                                              self.autoencoder_num_layers,
                                              self.is_cuda_avail,
                                              self.bidirectional)

        self.shared_linear = nn.Linear(self.autoencoder_bottleneck_feature_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()
        self.shared_layer_dropout = nn.Dropout(p=self.shared_layer_dropout_prob)
        self.shared_linear_1 = nn.Linear(self.shared_hidden_layer_size, self.shared_hidden_layer_size // 2)
        self.shared_activation_1 = nn.ReLU()

        self.user_head = nn.Sequential(nn.Linear(self.shared_hidden_layer_size // 2, self.user_dense_layer_hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(self.user_dense_layer_hidden_size, self.num_classes))

    def forward(self, user, input_seq, covariate_data=None):
        """
        Slightly complex forward pass. The autoencoder part return the decoded output
        which needs to be trained using MAE or MSE. The user head returns a vector of
        class probability distributions which need to be trained using cross entropy.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data: The covariates which will be concatenated with the output
        of the autoencoders before being used for classification.
        @return: output of the autoencoder and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        autoencoder_out = self.autoencoder(input_seq)
        bottle_neck = self.autoencoder.get_bottleneck_features(input_seq)
        bottle_neck = bottle_neck[:, -1, :]

        if covariate_data is not None:
            bottle_neck = torch.cat((bottle_neck, covariate_data.unsqueeze(0)), dim=1)

        shared_hidden_state = self.shared_linear(bottle_neck)
        shared_hidden_state = self.shared_activation(shared_hidden_state)
        # shared_hidden_state = self.shared_layer_dropout(shared_hidden_state)
        shared_hidden_state_1 = self.shared_linear_1(shared_hidden_state)
        shared_hidden_state_1 = self.shared_activation_1(shared_hidden_state_1)

        y_out = self.user_head(shared_hidden_state_1)

        return autoencoder_out, y_out
