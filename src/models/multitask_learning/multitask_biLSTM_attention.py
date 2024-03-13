import torch
import torch.nn as nn

from src.models.attention import attention, align_functions
from src.models import user_dense_heads
from src.bin import validations

ALIGNMENT_TYPE_FUNCTION_MAP = {
    'additive': align_functions.AdditiveAlignment,
    'general': align_functions.GeneralAlignment,
    'dot': align_functions.DotAlignment
}


class MultiTaskBiLSTMAttention(nn.Module):
    def __init__(self,
                 users: list,
                 lstm_input_size,
                 lstm_hidden_size,
                 lstm_num_layers,
                 covariate_hidden_size,
                 shared_hidden_layer_size,
                 user_dense_layer_hidden_size,
                 num_classes,
                 dropout=0,
                 num_covariates=0,
                 alignment_type="additive",
                 forward_pass_type="covariate_concat"):
        """
        This model has a dense layer for each student. This is used for MultiTask learning.

        @param users: List of students (their ids) that are going to be used for trained.
        @param lstm_input_size: Input size of the time series portion on the model.
        @param lstm_hidden_size: Hidden size of the LSTM.
        @param lstm_num_layers: Num layers in LSTM.
        @param user_dense_layer_hidden_size: dense head hidden size.
        @param num_classes: Number of classes in classification.
        @param num_covariates: Number of covariates to be concatenated to the dense layer before
                           generating class probabilities.
        """
        super(MultiTaskBiLSTMAttention, self).__init__()
        validations.check_if_element_in_list(forward_pass_type, ALLOWED_FORWARD_PASSES)
        validations.check_if_key_present_in_dict(alignment_type, ALIGNMENT_TYPE_FUNCTION_MAP)

        # Check if cuda available. If so set this flag true.
        self.is_cuda_avail = True if torch.cuda.device_count() > 0 else False
        self.users = users
        self.lstm_input_size = lstm_input_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.shared_hidden_layer_size = shared_hidden_layer_size
        self.user_dense_layer_hidden_size = user_dense_layer_hidden_size
        self.num_classes = num_classes
        self.num_covariates = num_covariates
        self.covariate_hidden_size = covariate_hidden_size

        # Layer initialization.
        self.biLSTM_attention_encoder = attention.AttentionEncoder(self.lstm_input_size,
                                                                   self.lstm_hidden_size,
                                                                   self.lstm_num_layers,
                                                                   self.is_cuda_avail,
                                                                   self.dropout)
        self.encoder_outputs_size, self.context_vector_size = self.biLSTM_attention_encoder.get_encoder_dimensions()

        self.alignment_function = ALIGNMENT_TYPE_FUNCTION_MAP[alignment_type](self.encoder_outputs_size,
                                                                              self.context_vector_size)
        self.attention = attention.Attention(self.alignment_function)

        # This is the layer to be used to calculate the expected vector, the attention layers are just initializations.
        # Return the fixed vector that we want to send through the shared layer.
        self.expected_vector_from_attention = attention.ExpectedContextVectorAfterAttention(self.attention)

        # This layer is used to return a vector of encoder output size.
        self.covariate_linear = nn.Linear(self.num_covariates, self.covariate_hidden_size)

        self.shared_linear = nn.Linear(self.lstm_hidden_size + self.num_covariates,
                                       self.shared_hidden_layer_size)

        self.shared_activation = nn.ReLU()

        self.user_heads = user_dense_heads.UserDenseHead(self.users,
                                                         self.shared_hidden_layer_size,
                                                         self.user_dense_layer_hidden_size,
                                                         self.num_classes)

        FORWARD_PASS_TYPE_FUNCTION_MAPS = {
            'covariate_concat': self.covariate_concat_forward,
            'covariate_hidden': self.covariate_hidden_forward
        }

        self.sequence_forward_pass = FORWARD_PASS_TYPE_FUNCTION_MAPS[forward_pass_type]

    def forward(self, user, input_seq, covariate_data=None):
        """
        The input sequence is inputed to the LSTM and the last hidden state of the LSTM
        is passed to the shared layer of the MultiTask Learner.

        @param user: The student for which the model is being trained. All the students
        contribute towards the loss of the auto encoder, but each have a separate linear
        head.
        @param input_seq: Must contain the input sequence that will be used to train the
        autoencoder.
        @param covariate_data(batch_size, num_covariates): The covariates which will be concatenated with the output
        of the autoencoders before being used for classification.
        @return: output of the autoencoder and the probability distribution of each class
        for the student.
        """
        validations.validate_integrity_of_covariates(self.num_covariates, covariate_data)

        encoder_outputs, context_vector = self.biLSTM_attention_encoder(input_seq)
        # encoder_outputs = [batch_size, seq_len, encoder_output_size]
        # context_vector = [batch_size, seq_len, context_vector_size]

        expected_vector = self.sequence_forward_pass(encoder_outputs, context_vector, covariate_data)

        return self.shared_layer_forward(expected_vector, user)

    def shared_layer_forward(self, expected_vector, user):
        shared_hidden_state = self.shared_linear(expected_vector)
        shared_hidden_state = self.shared_activation(shared_hidden_state)

        y_out = self.user_heads(user, shared_hidden_state)

        return y_out

    def covariate_concat_forward(self, encoder_outputs, context_vector, covariate_data):
        if covariate_data is not None:
            covariate_hidden = self.covariate_linear(covariate_data)
            # covariate_hidden = [batch_size, encoder_output_size]
            encoder_outputs = torch.cat((encoder_outputs, covariate_hidden), dim=1)
            # covariate_hidden = [batch_size, seq_len + 1, encoder_output_size]

        expected_vector = self.expected_vector_from_attention(encoder_outputs, context_vector)
        # expected_vector = [batch_size, 1, encoder_output_size]

        return expected_vector

    def covariate_hidden_forward(self, encoder_outputs, context_vector, covariate_data):
        """
        @brief: Uses the covariate_hidden layer as the hidden state to calculate additive attention.
                Essentially, the hidden layer of the covariate Linear layer will be used to calculate the score with
                the encoder outputs.
        """
        assert covariate_data is not None, "Covariates cannot be None for this forward pass."
        covariate_hidden = self.covariate_linear(covariate_data)
        # covariate_hidden = [batch_size, encoder_output_size]
        expected_vector = self.expected_vector_from_attention(encoder_outputs, covariate_hidden)

        return expected_vector
