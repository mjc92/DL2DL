import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=30, embed_size=128, hidden_size=512):
        super(Seq2Seq, self).__init__()
        self.vocab_size=30
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def encode(self, input):
        """
        Args:
            input: input tensor of shape (batch_size, seq_length)
        
        Returns:
            h: last hidden state of shape (num_layer*bidirectional, batch_size, hidden_size)
            c: last cell state of shape (num_layer*bidirectional, batch_size, hidden_size)
        """
        embedded = self.embed(input)          # (batch_size, seq_length, embed_size)
        _, (h, c) = self.encoder(embedded)   # (1, batch_size, hidden_size)
        return h, c
    
    def decode(self, input, h, c):
        """
        Args:
            input: decoder input (teacher forcing) of shape (batch_size, seq_length)
            h: last hidden state from the encoder
            c: last cell state from the encoder
            
        Returns:
            output: decoder outputs of shape (batch_size, seq_length, vocab_size)
        """
        embedded = self.embed(input)
        output, _ = self.decoder(embedded, (h, c))
        
        # Linear transform
        seq_length = output.size(1)
        output = output.contiguous().view(-1, output.size(2))
        output = self.linear(output).view(-1, seq_length, self.vocab_size)  # (batch_size, seq_length, vocab_size) 
        return output