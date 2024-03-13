import torch
import torch.nn as nn

from src.models.user_classifier import UserClassifier
from src.models import user_dense_heads
from src.bin import validations


class MultiTaskLSTM(nn.Module):
    def __init__(self,
                 users: list,
                 lstm_input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 lstm_bidirectional,
                 multitask_input_size,
                 multitask_hidden_size,
                 multitask_num_classes,
                 dropout=0,
                 num_covariates=0,
                 **shared_layer_params):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param lstm_input_size: Input size of the time series portion on the model.
        @param lstm_hidden_size: Hidden size of the LSTM.
        @param lstm_num_layers: Num layers in LSTM.
        @param lstm_bidirectional: LSTM is bidirectional if set True.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskLSTM, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_input_size = lstm_input_size

        if self.lstm_bidirectional:
            self.lstm_hidden_size = lstm_hidden_size // 2
        else:
            self.lstm_hidden_size = lstm_hidden_size

        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.num_covariates = num_covariates

        # Layer initialization.
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            bidirectional=self.lstm_bidirectional,
                            dropout=self.dropout)

        self.relu = nn.ReLU()

        self.user_classifier = UserClassifier(users,
                                              multitask_input_size + self.num_covariates,
                                              multitask_hidden_size,
                                              multitask_num_classes,
                                              dropout,
                                              **shared_layer_params)

    def forward(self, user, input_seq, covariate_data=None):
        """
        The input sequence is send to the LSTM and the last hidden state of the LSTM
        is passed to the multi classifier.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        LSTM.
        @param covariate_data: The covariates which will be concatenated with the output
        of the LSTM before being used for classification.
        @return: probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        lstm_out = self.lstm(input_seq)
        lstm_last_hidden_state = lstm_out[:, -1, :]
        lstm_last_hidden_state = self.relu(lstm_last_hidden_state)

        if covariate_data is not None:
            embedding = torch.cat((lstm_last_hidden_state, covariate_data.unsqueeze(0)), dim=1)

        y_out = self.user_classifier(user, embedding)

        decoded_sequence = None
        return decoded_sequence, y_out


class MultiTaskMultiLSTMLearner(nn.Module):
    def __init__(self,
                 users: list,
                 lstm_input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 lstm_bidirectional,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 num_covariates=0,
                 lstm_dropout_prob=0,
                 shared_layer_dropout_prob=0,
                 user_head_dropout_prob=0):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param lstm_input_size: Input size of the time series portion on the model.
        @param lstm_hidden_size: Hidden size of the LSTM.
        @param lstm_num_layers: Num layers in LSTM.
        @param lstm_bidirectional: LSTM is bidirectional if set True.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskMultiLSTMLearner, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_input_size = lstm_input_size
        self.lstm_dropout_prob = lstm_dropout_prob
        self.shared_layer_dropout_prob = shared_layer_dropout_prob
        self.user_head_dropout_prob = user_head_dropout_prob
        self.lstm_num_layers = lstm_num_layers
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates

        # Layer initialization.
        self.user_lstm = user_dense_heads.UserLSTM(input_size=self.lstm_input_size,
                                                   lstm_hidden_size=self.lstm_hidden_size,
                                                   num_layers=self.lstm_num_layers,
                                                   bidirectional=self.lstm_bidirectional,
                                                   dropout=self.lstm_dropout_prob)

        self.shared_linear = nn.Linear(self.lstm_hidden_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()
        self.shared_layer_dropout = nn.Dropout(p=self.shared_layer_dropout_prob)

        self.user_heads = user_dense_heads.UserDenseHead(self.users,
                                                         self.shared_hidden_layer_size,
                                                         self.user_dense_layer_hidden_size,
                                                         self.num_classes)

    def forward(self, user, input_seq, covariate_data=None):
        """
        The input sequence is inputed to the LSTM user head and the last hidden state of the LSTM
        is passed to the shared layer of the MultiTask Learner.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data: The covariates which will be concatenated with the output
        of the LSTM before being used for classification.
        @return: output of the LSTM and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        lstm_out = self.user_lstm(user, input_seq)
        lstm_last_hidden_state = lstm_out[:, -1, :]

        if covariate_data is not None:
            embedding = torch.cat((lstm_last_hidden_state, covariate_data.unsqueeze(0)), dim=1)

        shared_hidden_state = self.shared_linear(embedding)
        shared_hidden_state = self.shared_activation(shared_hidden_state)
        shared_hidden_state = self.shared_layer_dropout(shared_hidden_state)

        y_out = self.user_heads(user, shared_hidden_state)

        return y_out
