"""
Implementation is based on the paper -  NEURAL MACHINE TRANSLATION
BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
Author - Bahdanau et. al.

Deviation from paper - Use of LSTM rather than GRU.

This module is referred from -
https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
"""
import torch
from torch import nn

from src.models.attention import helper


class AttentionEncoder(nn.Module):
    """
    The RNN unit used here is LSTM.
    todo(abhinavshaw): Make the RNN configurable.
    """

    def __init__(self,
                 input_size,
                 encoder_hidden_size,
                 num_layers,
                 is_cuda,
                 dropout_p=0):
        super(AttentionEncoder, self).__init__()
        # Input size is same as feature size.
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.is_cuda = is_cuda
        self.dropout_p = dropout_p

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.encoder_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True)

        self.linear = nn.Linear(self.encoder_hidden_size * 2, self.encoder_hidden_size)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input_seq):
        """

        @param input_seq(batch_size,seq_len,input_size): The input sequence that needs to be encoded.
        @return: Encoded input sequence and the context vector which
        is formed by the last hidden state of the RNN.
        @attention: The last hidden state (both the forward and the backward hidden states)
                    are put through a fully connected layer with tanh activation(As in paper).
        """

        outputs, (hidden_state, cell_state) = self.rnn(input_seq)
        # outputs(seq_len, num_features)
        # hidden_state(2, sequence_len, hidden_size): Last hidden state of the LSTM

        # Hidden states are indexed as -1 and -2 because, the hidden state contains the forward
        # which is the first and backward hidden state which is the second state.
        hidden_forward, hidden_backward = hidden_state[-2, :, :], hidden_state[-1, :, :]
        combined_hidden_state = torch.cat((hidden_forward, hidden_backward), dim=1)
        # note: Check this linear layer here. This is deviation from the paper. In might be wrong.
        context_vector = torch.tanh(self.linear(combined_hidden_state))

        return outputs, context_vector

    def get_encoder_dimensions(self):
        encoder_output_size = self.encoder_hidden_size * 2
        context_vector_size = self.encoder_hidden_size

        return encoder_output_size, context_vector_size


class Attention(nn.Module):
    """
    A wrapper class to calculate the attention weights through softmax.
    """

    def __init__(self, align_function):
        super(Attention, self).__init__()
        # The align function must always return a tensor of size [batch_size, seq_len]
        self.align_function = align_function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, decoder_hidden_state):
        """

        @param encoder_outputs(batch_size, seq_len, encoder_output_dim): The encoder_outputs that need to be used to calculate the energy
        between the last hidden state of the decoder.

        @param decoder_hidden_state(batch_size, context_vector_dim): Last hidden state of the decoder. It is the last hidden state of the encoder at t=0.
        (initially)
        @return: Return the weights that need to be used to calculate the expected context vector.
        """

        attention = self.align_function(encoder_outputs, decoder_hidden_state)

        assert len(attention.shape) == 2, "Attention tensor must have exactly two dimensions, being of the shape [batch_size, seq_len]"
        b_s, s_len, encoder_h_s = helper.get_dimensions_from_encoder_outputs(encoder_outputs)
        attention__b_s, attention_s_len = attention.shape
        assert (b_s == attention__b_s) and (attention_s_len == s_len), "Attention vector must have shape of [batch_size, seq_len]."

        attention_weights = self.softmax(attention)

        return attention_weights


class ExpectedContextVectorAfterAttention(nn.Module):
    """
    This class encapsulates the procedure for calculating the expected vector from the attention weights and the
    sequence of encoder outputs.
    """

    def __init__(self, attention):
        """

        @param attention: The attention that needs to be used to calculate the expected vector.
        """
        super(ExpectedContextVectorAfterAttention, self).__init__()
        self.attention = attention

    def forward(self, encoder_outputs, previous_decoder_hidden_state):
        """
        @param previous_decoder_hidden_state(batch_size, decoder_hidden_size): Hidden state of the Decoder at t-1. In the first pass this
        will be the last hidden state of the Encoder (context vector)
        @param encoder_outputs(batch_size, seq_len, encoder_hidden_dim * 2): The sequence of Encoder ouputs.
        @return: expected_vector from encoder hidden states after attention. Shape (batch_size, 1, encoder_hidden_dim * 2)
        """

        attention_weights = self.attention(encoder_outputs, previous_decoder_hidden_state)
        # attention weights = [batch_size,  seq_len]
        attention_weights = attention_weights.unsqueeze(1)
        # attention weights = [batch_size, 1,  seq_len]
        expected_encoder_output = torch.bmm(attention_weights, encoder_outputs)
        # expected_encoder_output = [batch_size, 1, encoder_output_size], this is also the weighted encoder output.

        return expected_encoder_output


class AttentionDecoder(nn.Module):
    """
    This class is AttentionDecoder.
    """

    def __init__(self,
                 output_size,
                 encoder_output_size,
                 decoder_hidden_size,
                 attention,
                 input_size,
                 context_vector_size=None,
                 dropout_p=0):
        """
        @param output_size: This is equal to input size if the Encoder-Decoder pair is used for dimensionality reduction.
        @param context_vector_size: This is just accepted to assert the correct size of the decoder hidden state.
        @param attention: Attention model to be used for the decoder.
        """
        super(AttentionDecoder, self).__init__()
        # The encoder_output_size should be equal to the decoder_hidden_state
        # because this is what is used as the first hidden state in the decoder.
        # This is used to calculate the expected context vector.
        if context_vector_size is not None:
            assert context_vector_size == decoder_hidden_size, "context_vector size not equal to the decoder_hidden size."

        self.output_size = output_size
        self.encoder_output_size = encoder_output_size
        self.decoder_hidden_size = decoder_hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p

        self.expected_vector_from_attention = ExpectedContextVectorAfterAttention(attention)

        # Decoder rnn. LSTM for now, but this could be changed to any RNN.
        self.rnn = nn.LSTM(input_size=self.input_size + self.encoder_output_size,
                           hidden_size=self.decoder_hidden_size,
                           batch_first=True)
        # This layer does a transform to a concatenated [encoder_outputs, decoder_rnn_output, input(target)]
        self.output = nn.Linear(self.encoder_output_size + self.decoder_hidden_size + self.input_size, self.output_size)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, input_vector, encoder_outputs, previous_decoder_hidden_state):
        """
        @param input_vector(batch_size, 1, input_size): The input vector that needs to be translated.
        @param previous_decoder_hidden_state(batch_size, decoder_hidden_size): Hidden state of the Decoder at t-1. In the first pass this
        will be the last hidden state of the Encoder (context vector)
        @param encoder_outputs(batch_size, seq_len, encoder_hidden_dim * 2): The sequence of Encoder ouputs.

        @return: The predicted target that can be used to minimize the reconstruction loss.
        """

        expected_encoder_output = self.expected_vector_from_attention(encoder_outputs, previous_decoder_hidden_state)
        rnn_input = torch.cat((input_vector, expected_encoder_output), dim=2)
        # rnn_input = [batch_size, 1, input_size + encoder_output_size]

        rnn_output, (hidden, cell) = self.rnn(rnn_input)
        # rnn_output = [batch_size, seq_len, input_vector_seq_len, decoder_hidden_size] seq_len is 1, while decoding
        # since the input is a vector and not a sequence.

        assert (hidden ==
                rnn_output).all(), "Since, n directions and num layers is always 1 the hidden state and output should be the same."

        output = self.output(torch.cat((rnn_output, input_vector, expected_encoder_output), dim=2))
        # output = [batch_size, 1, output_dim]

        return output, hidden.squeeze(1)
