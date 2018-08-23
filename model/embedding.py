import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, seq_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * self.div_term)
        self.pe[:, 1::2] = torch.cos(position * self.div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x, t):
        x = x + self.pe[:, :x.size(1)]
        pe = t * self.div_term
        x = x + pe.sin() if t % 2 == 0 else pe.cos()
        return x

#
# class SentenceEmbedding(nn.Module):
#     def __init__(self, seq_len):
#         super().__init__()
#         self.seq_len = seq_len
#         self.weight = nn.Parameter(torch.tensor(seq_len))
#

