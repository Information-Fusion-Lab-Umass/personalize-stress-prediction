import torch
import torch.nn as nn

from src.models import autoencoder
from src.models import user_dense_heads
from src.bin import validations
from src.utils import object_generator_utils as object_generator


class MultiTaskAutoEncoderLearner(nn.Module):
    def __init__(self,
                 users: list,
                 groups: dict, 
                 num_branches,
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
                 train_only_with_covariates=False,
                 bidirectional=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param autoencoder_input_size: Input size of the time series portion on the model.
        @param autoencoder_bottleneck_feature_size: Encoded input size of autoecoder.
        @param autoencoder_num_layers: Num layers in autoencoder LSTM model.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskAutoEncoderLearner, self).__init__()

        if train_only_with_covariates:
            assert num_covariates > 0, "The model has to be provided either input sequence or covariates."

        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.groups = groups
        self.num_branches = num_branches
        self.autoencoder_input_size = autoencoder_input_size
        # Ignore the autoencoder input feature if you are just training on sequences.
        self.autoencoder_bottleneck_feature_size = autoencoder_bottleneck_feature_size if not train_only_with_covariates else 0
        self.autoencoder_num_layers = autoencoder_num_layers
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates
        self.shared_layer_dropout_prob = shared_layer_dropout_prob
        self.user_head_dropout_prob = user_head_dropout_prob
        self.ordinal_regression_head = ordinal_regression_head
        self.train_only_with_covariates = train_only_with_covariates
        self.bidirectional = bidirectional

        # Layer initialization.
        if not train_only_with_covariates:
            self.autoencoder = autoencoder.LSTMAE(self.autoencoder_input_size,
                                                  self.autoencoder_bottleneck_feature_size,
                                                  self.autoencoder_num_layers,
                                                  self.is_cuda_avail,
                                                  self.bidirectional)

        # self.shared_linear = nn.Linear(self.autoencoder_bottleneck_feature_size + self.num_covariates,
        #                                self.shared_hidden_layer_size)
        # self.shared_activation = nn.ReLU()
        # self.shared_layer_dropout = nn.Dropout(p=self.shared_layer_dropout_prob)
        # self.shared_linear_1 = nn.Linear(self.shared_hidden_layer_size, self.shared_hidden_layer_size // 2)
        # self.shared_activation_1 = nn.ReLU()

        # # Original SOTA module
        # self.user_heads = user_dense_heads.UserDenseHead(self.users,
        #                                                  self.shared_hidden_layer_size // 2,
        #                                                  self.user_dense_layer_hidden_size,
        #                                                  self.num_classes,
        #                                                  self.user_head_dropout_prob,
        #                                                  self.ordinal_regression_head)

        # # Tried module in Spring 2020 (by yunfeiluo)
        # self.user_heads = user_dense_heads.GroupDenseHead(self.groups,
        #                                                  self.shared_hidden_layer_size // 2,
        #                                                  self.user_dense_layer_hidden_size,
        #                                                  self.num_classes,
        #                                                  self.user_head_dropout_prob,
        #                                                  self.ordinal_regression_head)

        # # New Try in Fall 2020 (by yunfeiluo), with students' head
        # self.user_heads = user_dense_heads.BranchingUserDenseHead(
        #     self.users,
        #     self.shared_hidden_layer_size // 2,
        #     self.user_dense_layer_hidden_size,
        #     self.user_dense_layer_hidden_size,
        #     self.num_classes,
        #     self.num_branches,
        #     self.user_head_dropout_prob,
        #     self.ordinal_regression_head
        # )

        # # without students' head, end after branching layers
        # self.user_heads = user_dense_heads.BranchingDenseHead(
        #     self.users,
        #     self.shared_hidden_layer_size // 2,
        #     self.user_dense_layer_hidden_size,
        #     self.num_classes,
        #     self.num_branches,
        #     self.user_head_dropout_prob,
        #     self.ordinal_regression_head
        # )

        # # without shared dense layer, but with user heads
        # self.user_heads = user_dense_heads.BranchingUserDenseHead(
        #     self.users,
        #     self.autoencoder_bottleneck_feature_size + self.num_covariates,
        #     self.shared_hidden_layer_size,
        #     self.user_dense_layer_hidden_size,
        #     self.num_classes,
        #     self.num_branches,
        #     self.user_head_dropout_prob,
        #     self.ordinal_regression_head
        # )

        # curr best on 5-fold
        # without shared dense layer, but with user heads
        self.user_heads = user_dense_heads.BranchingUserBlock(
            self.users,
            self.autoencoder_bottleneck_feature_size + self.num_covariates,
            self.shared_hidden_layer_size,
            self.user_dense_layer_hidden_size,
            self.num_classes,
            self.num_branches,
            self.user_head_dropout_prob,
            self.ordinal_regression_head
        )

        # # with shared dense layer, with user heads
        # self.user_heads = user_dense_heads.BranchingUserBlock(
        #     self.users,
        #     self.shared_hidden_layer_size // 2,
        #     self.shared_hidden_layer_size // 2,
        #     self.user_dense_layer_hidden_size,
        #     self.num_classes,
        #     self.num_branches,
        #     self.user_head_dropout_prob,
        #     self.ordinal_regression_head
        # )

    def forward(self, user, branch_id, input_seq, covariate_data=None, shared_out=None, has_shared=False):
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

        # return if shared_out is provided
        # if has_shared:
        #     return self.user_heads(user, branch_id, shared_out)
        
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)
        # If not training on sequences, do not put the sequences through he auto encoder.
        if not self.train_only_with_covariates:
            autoencoder_out = self.autoencoder(input_seq)
            bottle_neck = self.autoencoder.get_bottleneck_features(input_seq)
            bottle_neck = bottle_neck[:, -1, :]
        else:
            bottle_neck = object_generator.get_tensor_on_correct_device([])

        if covariate_data is not None:
            bottle_neck = torch.cat((bottle_neck, covariate_data.unsqueeze(0)), dim=1)

        # # with shared dense layer
        # shared_hidden_state = self.shared_linear(bottle_neck)
        # shared_hidden_state = self.shared_activation(shared_hidden_state)
        # shared_hidden_state = self.shared_layer_dropout(shared_hidden_state)
        # shared_hidden_state_1 = self.shared_linear_1(shared_hidden_state)
        # shared_hidden_state_1 = self.shared_activation_1(shared_hidden_state_1)
        
        # shared_out = shared_hidden_state_1
        shared_out = bottle_neck

        # y_out = self.user_heads(user, branch_id, shared_out)
        y_out = self.user_heads(user, shared_out)

        # return autoencoder_out if not self.train_only_with_covariates else None, y_out, shared_out
        return autoencoder_out if not self.train_only_with_covariates else None, y_out
