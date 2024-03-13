import torch
import torch.nn as nn
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input_seq):
        tt = torch.cuda if self.isCuda else torch
        # h0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, input_seq.size(0), self.hidden_size))
        # c0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, input_seq.size(0), self.hidden_size))

        encoded_input, hidden = self.lstm(input_seq)
        encoded_input = self.relu(encoded_input)
        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        """

        @param hidden_size: Hidden size is the size of the encoded input, usually encoder hidden size.
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda

        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.output_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        # h0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        # c0 = torch.autograd.Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=False, bidirectional=False):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size,
                                  hidden_size,
                                  num_layers,
                                  isCuda,
                                  bidirectional)
        self.decoder = DecoderRNN(hidden_size,
                                  input_size,
                                  num_layers,
                                  isCuda)

    def forward(self, input_seq):
        encoded_input = self.encoder(input_seq)
        decoded_output = self.decoder(encoded_input)
        return decoded_output

    def get_bottleneck_features(self, input_seq):
        return self.encoder(input_seq)
