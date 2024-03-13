import warnings
import torch
import torch.nn as nn
import numpy as np

from src.definitions import LOW_MODEL_CAPACITY_WARNING

class softmax_select(nn.Module):
    def __init__(self, num_branches):
        super(softmax_select, self).__init__()
        self.num_branches = num_branches
        self.prob = nn.Parameter(torch.exp(torch.ones(self.num_branches, device=torch.device("cuda"))), requires_grad=True)
        self.T = 10.0

    def forward(self):
        eps = torch.rand(self.num_branches, device=torch.device("cuda"))
        log_prob = torch.log(self.prob)
        if self.training:
            log_prob = (log_prob + eps) / self.T
            return torch.exp(log_prob[torch.argmax(log_prob)]) / torch.exp(log_prob).sum(), str(torch.argmax(log_prob).tolist())
        return 1.0, str(torch.argmax(log_prob).tolist())

class BranchingUserBlock(nn.Module):
    def __init__(self, users: list, input_size, branch_hidden_size, user_hidden_size, num_classes, num_branches, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(BranchingUserBlock, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.branch_hidden_size = branch_hidden_size
        self.user_hidden_size = user_hidden_size
        self.num_classes = num_classes
        self.num_branches = num_branches
        self.dropout = dropout

        # Layer initialization.
        if self.input_size > self.branch_hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        
        # Construct dictionary for branching layer
        branching_layer = dict()
        for i in range(self.num_branches):
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.branch_hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=0.5),
                nn.Linear(self.branch_hidden_size, self.branch_hidden_size // 2),
                nn.ReLU()
            )
            
            branching_layer[str(i)] = sequential_liner
        
        self.branching_layer = nn.ModuleDict(branching_layer)

        # construct user layers with probability distribution
        user_layer = dict()
        branching_probs = dict()
        for user in users:
            sequential_liner = nn.Sequential(
                nn.Linear(self.branch_hidden_size // 2, self.user_hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=0.5),
                nn.Linear(self.user_hidden_size, self.num_classes))

            if ordinal_regression_head:
                sequential_liner.add_module("sigmoid", nn.Sigmoid())
            
            user_layer[user] = sequential_liner

            prob_module = softmax_select(self.num_branches)
            branching_probs[user] = prob_module

        self.user_layer = nn.ModuleDict(user_layer)
        self.branching_probs = nn.ModuleDict(branching_probs)

    def forward(self, user, input_data):
        prob_out, branch_ind = self.branching_probs[user]()
        branching_out = prob_out * self.branching_layer[branch_ind](input_data)

        return self.user_layer[user](branching_out)

class BranchingDenseHead(nn.Module):
    def __init__(self, users: list, input_size, hidden_size, num_classes, num_branches, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(BranchingDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_branches = num_branches
        self.dropout = dropout

        # Layer initialization.
        if self.input_size > self.hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        
        # Construct dictionary for branching layer
        branching_layer = dict()
        for i in range(self.num_branches):
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.num_classes)
            )

            if ordinal_regression_head:
                sequential_liner.add_module("sigmoid", nn.Sigmoid())
            
            branching_layer[str(i)] = sequential_liner
        
        self.branching_layer = nn.ModuleDict(branching_layer)

    def forward(self, user, branch_id, input_data):
        # user is not used here. Leav if here for the purpose of not modifying to much on the other files. 
        return self.branching_layer[branch_id](input_data)

class BranchingUserDenseHead(nn.Module):
    def __init__(self, users: list, input_size, branch_hidden_size, user_hidden_size, num_classes, num_branches, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(BranchingUserDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.branch_hidden_size = branch_hidden_size
        self.user_hidden_size = user_hidden_size
        self.num_classes = num_classes
        self.num_branches = num_branches
        self.dropout = dropout

        # Layer initialization.
        if self.input_size > self.branch_hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        
        # Construct dictionary for branching layer
        branching_layer = dict()
        for i in range(self.num_branches):
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.branch_hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=dropout),
                nn.Linear(self.branch_hidden_size, self.branch_hidden_size // 2),
                nn.ReLU()
            )
            
            branching_layer[str(i)] = sequential_liner
        
        self.branching_layer = nn.ModuleDict(branching_layer)

        # construct user layers
        dense_layer = dict()
        for user in users:
            sequential_liner = nn.Sequential(
                nn.Linear(self.branch_hidden_size // 2, self.user_hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=dropout),
                nn.Linear(self.user_hidden_size, self.num_classes))

            if ordinal_regression_head:
                sequential_liner.add_module("sigmoid", nn.Sigmoid())

            dense_layer[user] = sequential_liner

        self.student_dense_layer = nn.ModuleDict(dense_layer)

    def forward(self, user, branch_id, input_data):
        branching_out = self.branching_layer[branch_id](input_data)
        return self.student_dense_layer[user](branching_out)

class GroupDenseHead(nn.Module):
    def __init__(self, groups: dict, input_size, hidden_size, num_classes, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each group of students. This is used for MultiTask learning.

        @param groups: dictionary of groups of student, map: student_ids -> group_ids
        The ids of group and student much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(GroupDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.groups = groups # map: student -> group

        group_nodes = set()
        for student in groups:
            group_nodes.add(groups[student])

        # Layer initialization.
        if self.input_size > self.hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        dense_layer = dict()

        # make a dense layer for each group
        for group in group_nodes:
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.num_classes))
            
            dense_layer[group] = sequential_liner

        self.student_dense_layer = nn.ModuleDict(dense_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[self.groups[user]](input_data)
        
class UserDenseHead(nn.Module):
    def __init__(self, users: list, input_size, hidden_size, num_classes, dropout=0, ordinal_regression_head=False):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each dense layer.
        @param hidden_size: Hidden size of the dense layer.
        """
        super(UserDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        # Layer initialization.
        if self.input_size > self.hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        dense_layer = {}
        for user in users:
            # todo(abhinavshaw): Make this configurable to any model of the users choice. can take those layers as a list.
            sequential_liner = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                # nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.num_classes))

            if ordinal_regression_head:
                sequential_liner.add_module("sigmoid", nn.Sigmoid())

            dense_layer[user] = sequential_liner

        self.student_dense_layer = nn.ModuleDict(dense_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[user](input_data)


class UserLSTM(nn.Module):
    def __init__(self, users: list,
                 input_size,
                 lstm_hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0):
        """
        This model has a LSTM for each user layer for each student.
        This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        The student ids much be strings.
        @param input_size: Input size of each LSTM.
        @param lstm_hidden_size: Hidden size of the LSTM.
        """
        super(UserDenseHead, self).__init__()
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        if self.bidirectional:
            self.lstm_hidden_size = self.lstm_hidden_size // 2

        # Layer initialization.
        if self.input_size > self.lstm_hidden_size:
            warnings.warn(LOW_MODEL_CAPACITY_WARNING)
        lstm_layer = {}
        for user in users:
            # todo(abhinavshaw): Make this configurable to any model of the users choice. can take those layers as a list.
            lstm_layer[user] = nn.LSTM(input_size=input_size,
                                       hidden_size=self.lstm_hidden_size,
                                       batch_first=True,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=dropout)

        self.student_dense_layer = nn.ModuleDict(lstm_layer)

    def forward(self, user, input_data):
        return self.student_dense_layer[user](input_data)
