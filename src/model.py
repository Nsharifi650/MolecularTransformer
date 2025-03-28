import torch
import torch.nn as nn
import torch.nn.functional as F

from src.EncoderDecoder import EncoderLayer, Decoder
from src.config import Config


class Transformer(nn.Module):
    def __init__(self, config: Config, target_vocab_size: int):
        super(Transformer, self).__init__()

        # self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size)
        self.encoder = EncoderLayer(config.model.enc_d_model, config.model.enc_dff)

        dec_dff = config.model.enc_dff
        self.decoder = Decoder(
            config.model.num_layers,
            config.model.dec_d_model,
            config.model.dec_num_heads,
            dec_dff,
            target_vocab_size,
            config.model.pe_target,
        )

        self.final_layer = nn.Linear(config.model.dec_d_model, target_vocab_size)

    def forward(
        self,
        properties,
        target,
        look_ahead_mask,
        dec_padding_mask,
        training: bool = True,
    ):
        enc_output = self.encoder(properties)

        enc_output_reshaped = enc_output.unsqueeze(1).repeat(1, target.shape[1], 1)

        dec_output = self.decoder(
            target, enc_output_reshaped, look_ahead_mask, dec_padding_mask
        )
        ffl_output = self.final_layer(dec_output)

        #####during training:
        if training:
            return ffl_output
        else:
            ##### During inference::
            probabilities = F.softmax(ffl_output, dim=-1)
            predicted_tokens = torch.argmax(probabilities, dim=-1)
            return predicted_tokens
