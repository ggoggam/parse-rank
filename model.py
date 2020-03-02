from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size

        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_size,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            batch_first=True,
                            hidden_size=self.rnn_size,
                            num_layers=1)
        self.dense = nn.Linear(in_features=self.rnn_size,
                               out_features=self.vocab_size)

    def forward(self, inputs, lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (window_size, batch_size)
        :param lengths: array of actual lengths (no padding) of each input

        :return: the logits, a tensor of shape
                 (window_size, batch_size, vocab_size)
        """

        max_length = inputs.shape[-1]

        out = self.embedding(inputs)
        out = pack_padded_sequence(out, batch_first=True, enforce_sorted=False, lengths=lengths)
        out, (_, _) = self.lstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_length)
        out = self.dense(out)

        return out