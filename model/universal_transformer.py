import torch.nn as nn
import torch.nn.functional as fnn
import torch

from model.encoder import UTransformerEncoder
from model.decoder import UTransformerDecoder
from model.utils import subsequent_mask


class UniversalTransformer(nn.Module):
    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_enc_vocab, n_dec_vocab, h, t_steps=5, dropout=0.5,
                 sos_index=1):
        super().__init__()
        self.encoder = UTransformerEncoder(enc_seq_len, d_model, h, dropout)
        self.decoder = UTransformerDecoder(dec_seq_len, d_model, h, dropout)
        self.input_embed = nn.Embedding(n_enc_vocab, d_model, padding_idx=0)
        self.target_embed = nn.Embedding(n_dec_vocab, d_model, padding_idx=0)
        self.generator = nn.Linear(d_model, n_dec_vocab)

        self.t_steps = t_steps
        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.sos_index = sos_index

    def forward(self, source, target=None, source_mask=None, target_mask=None):
        batch_size, device = source.size(0), source.device
        source_mask = source_mask.unsqueeze(-2)

        x = self.input_embed(source)

        # Story Word Embedding Sum
        x = x.sum(dim=-2)

        for step in range(self.t_steps):
            x = self.encoder(x, step, source_mask)

        output_distribution = []
        decoder_input = torch.zeros(batch_size, 1).long().fill_(self.sos_index).to(device)

        for dec_step in range(self.dec_seq_len):
            target_mask = subsequent_mask(decoder_input.size(1)).to(device)
            y = self.input_embed(decoder_input)

            for step in range(self.t_steps):
                y = self.decoder(x, y, step, source_mask, target_mask)

            output = fnn.log_softmax(self.generator(y), dim=-1)

            word_idx = output[:, -1].argmax(dim=-1, keepdim=True)
            output_distribution.append(output[:, -1].unsqueeze(1))

            if target is None:  # decoder output -> next input
                decoder_input = torch.cat([decoder_input, word_idx], dim=-1)
            else:  # target_index -> next_input
                decoder_input = torch.cat([decoder_input, target[:, dec_step].unsqueeze(-1)], dim=-1)

        return torch.cat(output_distribution, dim=1)
