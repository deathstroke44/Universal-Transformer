import torch.nn as nn
import torch.nn.functional as fnn
import torch

from model.encoder import UTransformerEncoder
from model.decoder import UTransformerDecoder


class UniversalTransformer(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_enc_vocab, n_dec_vocab, h, t_steps=5, dropout=0.5,
                 sos_index=1):
        super().__init__()
        self.encoder = UTransformerEncoder(enc_seq_len, d_model, h, dropout)
        self.decoder = UTransformerDecoder(dec_seq_len, d_model, h, dropout)
        self.input_embed = nn.Embedding(n_enc_vocab, d_model)
        self.target_embed = nn.Embedding(n_dec_vocab, d_model)
        self.generator = nn.Linear(d_model, n_dec_vocab)

        self.t_steps = t_steps
        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.sos_index = sos_index

    def forward(self, source, target=None):
        batch_size, device = source.size(0), source.device

        x = self.input_embed(source)
        x = [self.encoder(x, step) for step in range(self.t_steps)]

        output_distribution = []
        decoder_input = torch.zeros(source.size(0), 1).fill_(self.sos_index).to(device)

        for dec_step in range(self.dec_seq_len):
            decoder_input = target[:, :dec_step + 1] if target is not None else decoder_input
            y = self.target_embed(decoder_input)
            y = [self.decoder(x, y, step) for step in range(self.t_steps)]
            y = fnn.log_softmax(self.generator(y), dim=-1)[:, -1]

            word_idx = y.argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, word_idx], dim=-1)
            output_distribution.append(y)

        return torch.cat(output_distribution, dim=-1)
