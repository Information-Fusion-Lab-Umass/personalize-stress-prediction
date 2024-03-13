import torch
import torch.nn as nn

from src.models.attention import helper


class AdditiveAlignment(nn.Module):
    """
    This class is the align function for additive attention.
    --------------------------------------------------------------------------
    s_t: Initially, the last hidden state of the Encoder is the initial hidden state of the Decoder.
    This hidden state is the one that is used to calculate the energy between the Encoder outputs. (value a_ij)

    Note: aka concat attention.
    --------------------------------------------------------------------------
    Equation - score(s_t, h_i) = v_a_⊤ * tanh (W_a[s_t; h_i])
    """
    def __init__(self,  encoder_output_size, context_vector_size):
        super(AdditiveAlignment, self).__init__()
        self.encoder_output_size = encoder_output_size
        self.context_vector_size = context_vector_size

        # The scoring layer learns a function that calculates the score between the decoder hidden state and encoder output.
        self.score = nn.Linear(self.encoder_output_size + self.context_vector_size,
                               self.context_vector_size)

        # This matrix is multiplied by the energy which is a sequence of vectors of len [context_vector_dim]
        # To get a final vector of seq len, we need v to be [1, context_vector_dim, 1]
        # and the Energy to be [batch_size, seq_len, context_vector_dim]
        self.v = nn.Parameter(torch.rand(self.context_vector_size, 1), requires_grad=True)

    def forward(self, encoder_outputs, decoder_hidden_state):
        """

        @param encoder_outputs(batch_size, seq_len, encoder_output_dim): The encoder_outputs that need to be used to calculate the energy
        between the last hidden state of the decoder.

        @param decoder_hidden_state(batch_size, context_vector_dim): Last hidden state of the decoder. It is the last hidden state of the encoder at t=0.
        (initially)
        @return: Return the alignment function that will go through the softmax for the final attention weights.
        """

        # Preparing the hidden state to match the dimensions of the ended_outputs.
        batch_size, seq_len, encoder_hidden_size = helper.get_dimensions_from_encoder_outputs(encoder_outputs)
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)
        decoder_hidden_state = decoder_hidden_state.repeat(1, seq_len, 1)

        # Calculating energy between the hidden state and each encoder output.
        # concatenated = [batch_size, seq_len, encoder_output_dim + context_vector_dim]
        concatenated = torch.cat((decoder_hidden_state, encoder_outputs), dim=2)
        # energy = [batch_size, seq_len, context_vector_dim]
        energy = torch.tanh(self.score(concatenated))

        # v = [context_vector_dim, 1]
        v = self.v.unsqueeze(0).repeat(batch_size, 1, 1)
        # v = [batch_size, context_vector_dim, 1]
        attention = torch.bmm(energy, v).squeeze(2)
        # attention = [batch_size, seq_len]

        return attention


class GeneralAlignment(nn.Module):
    """
    This attention is based on the equation in paper - Effective Approaches to Attention-based Neural Machine Translation
    Luong et. al.
    --------------------------------------------------------------------------
    Equation for score function - s_t_T * W_a * h_i
    --------------------------------------------------------------------------
    """

    def __init__(self, encoder_output_size, context_vector_size):
        super(GeneralAlignment, self).__init__()
        self.encoder_output_size = encoder_output_size
        self.context_vector_size = context_vector_size

        self.score = nn.Linear(encoder_output_size, context_vector_size)

    def forward(self, encoder_outputs, decoder_hidden_state):
        """
        @param encoder_outputs(batch_size, seq_len, encoder_output_dim): The encoder_outputs that need to be used to calculate the energy
        between the last hidden state of the decoder.
        @param decoder_hidden_state(batch_size, context_vector_dim): Last hidden state of the decoder. It is the last hidden state of the encoder at t=0.
        (initially)

        @return(batch_size, seq_len): Return the weights that need to be used to calculate the expected context vector.
        """

        # Preparing the hidden state to match the dimensions of the ended_outputs.
        batch_size, seq_len, encoder_hidden_size = helper.get_dimensions_from_encoder_outputs(encoder_outputs)
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)
        # decoder_hidden_state = [batch_size, 1, context_vector_size]
        decoder_hidden_state = decoder_hidden_state.permute(0, 2, 1)

        attention = self.score(encoder_outputs)
        # attention = [batch_size, seq_len, context_vector_dim]
        attention = torch.bmm(attention, decoder_hidden_state).squeeze(2)
        # attention = [batch_size, seq_len]

        return attention


class DotAlignment(nn.Module):
    """
    DotAttention
    --------------------------------------------------------------------------
    Equation - s_t_T * h_i
    --------------------------------------------------------------------------
    """
    def __init__(self, encoder_output_size, context_vector_size):
        """
        @attention: Context_vecto_size must be same as encoder_output_size as to calculate the dot product between these
                    vectors we need them to be of the same dimension.
        """
        super(DotAlignment, self).__init__()
        assert (encoder_output_size == context_vector_size), "context_vector_size must be same as the encoder_output_size"
        self.encoder_output_size = encoder_output_size
        self.context_vector_size = context_vector_size

    def forward(self, encoder_outputs, decoder_hidden_state):
        """
        Parameter description same as other align function layers.
        """
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)
        print("decoder_hidden_state", decoder_hidden_state.shape)
        # decoder_hidden_state = [batch_size, 1, context_vector_size]
        decoder_hidden_state = decoder_hidden_state.permute(0, 2, 1)
        # decoder_hidden_state = [batch_size, context_vector_size, 1]
        print("decoder_hidden_state", decoder_hidden_state.shape)
        attention = torch.bmm(encoder_outputs, decoder_hidden_state).squeeze(2)

        return attention
