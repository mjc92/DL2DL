import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Summary2Seq(nn.Module):
    def __init__(self, vocab_size=30, embed_size=128, hidden_size=512):
        super(Summary2Seq, self).__init__()
        self.vocab_size=30
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
    def encode(self, input, input_lens):
        """
        Args:
            input: input tensor of shape (batch_size, seq_length)
            input_lens: list containing length of each sample, (batch_size)
        
        Returns:
            h: last hidden state of shape (num_layer*bidirectional, batch_size, hidden_size)
            c: last cell state of shape (num_layer*bidirectional, batch_size, hidden_size)
        """
        embedded = self.embed(input)          
        pack_embedded = pack(embedded, input_lens, True) # (padded - batch_size & seq_length, embed_size)
        _, (h, c) = self.encoder(pack_embedded)   # (1, batch_size, hidden_size)
        return h, c
    
    def decode(self, target, output_lens, h, c):
        """
        Args:
            input: decoder input (teacher forcing) of shape (batch_size, seq_length)
            h: last hidden state from the encoder
            c: last cell state from the encoder
            
        Returns:
            output: decoder outputs of shape (batch_size, seq_length, vocab_size)
        """
        embedded = self.embed(target)          
        pack_embedded = pack(embedded, output_lens, True) # (padded - batch_size & seq_length, embed_size)
        output, _ = self.decoder(pack_embedded, (h, c)) # (padded - batch_size & seq_length, hidden_size)
        
        # Linear transform
        output = self.linear(output[0]) # (batch_size, vocab_size)        
        
        return output
    
    def sample(self, input, h, c, max_length):
        """Samples sentence for given image features (Greedy search).
        
        Args:
            input: start tokens of shape (batch_size, 1)
            h: last hidden state from the encoder
            c: last cell state from the encoder
            max_length: maximum sampling length
        """
        sampled_ids = []
        input = self.embed(input) 
        for i in range(max_length):                               
            hiddens, (h, c) = self.decoder(input, (h, c))         # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))             # (batch_size, vocab_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            input = self.embed(predicted.unsqueeze(1))           # (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, dim=1)             # (batch_size, 120)
        return sampled_ids