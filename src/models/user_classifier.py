import torch
import torch.nn as nn

from src.models.user_dense_heads import UserDenseHead
from src.bin import validations


class UserClassifier(nn.Module):
    def __init__(self, users: list,
                 multitask_input_size,
                 multitask_hidden_size,
                 multitask_num_classes,
                 multitask_dropout=0,
                 **shared_layer_params):
        """
        This wraps multiple fully connected layers followed by the multitask layer for users.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param multitask_input_size: Input size of each dense layer.
        @param multitask_hidden_size: Hidden size of the dense layer.
        @param multitask_num_classes: Number of classes that the multitask layer will output.
        @param multitask_dropout: dropout for multitask head.
        """
        super(UserClassifier, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False

        # Extracting params for the shared layer.
        self.sl_input_size = shared_layer_params.get("sl_input_size", )
        self.sl_hidden_size = shared_layer_params.get("sl_hidden_size", )
        self.sl_dropout_prob = shared_layer_params.get("sl_dropout_prob", )
        self.sl_1_input_size = shared_layer_params.get("sl_1_input_size", )
        self.sl_1_hidden_size = shared_layer_params.get("sl_1_hidden_size", )
        self.sl_1_dropout_prob = shared_layer_params.get("sl_1_dropout_prob", )

        self.users = users
        self.multitask_input_size = multitask_input_size
        self.multitask_hidden_size = multitask_hidden_size
        self.multitask_num_classes = multitask_num_classes
        self.multitask_dropout = multitask_dropout

        validations.validate_sequential_model_size_parameters(self.sl_input_size,
                                                              self.sl_hidden_size,
                                                              self.sl_1_input_size,
                                                              self.sl_1_hidden_size,
                                                              self.multitask_input_size,
                                                              self.multitask_hidden_size)

        self.sl = nn.Linear(self.sl_input_size, self.sl_hidden_size)
        self.sl_activation = nn.ReLU()
        self.sl_dropout = nn.Dropout(p=self.sl_dropout)

        self.sl_1 = nn.Linear(self.sl_1_input_size, self.sl_1_hidden_size)
        self.sl_1_activation = nn.ReLU()
        self.sl_1_dropout = nn.Dropout(p=self.sl_1_dropout_prob)

        self.user_dense_head = UserDenseHead(self.users,
                                             self.multitask_input_size,
                                             self.multitask_hidden_size,
                                             self.multitask_num_classes,
                                             self.multitask_dropout)

    def forward(self, user, input_data):

        sl_out = self.sl(input_data)
        sl_out = self.sl_activation(sl_out)
        sl_out = self.sl_dropout(sl_out)

        sl_1_out = self.sl_1(sl_out)
        sl_1_out = self.sl_1_activation(sl_1_out)
        sl_1_out = self.sl_1_dropout(sl_1_out)

        return self.student_dense_layer[user](sl_1_out)
