import torch
import torch.nn as nn
import numpy as np

from src.multihead import MultiHeadAttention
from utils.logging_config import get_logger

logger = get_logger(__name__)


# THE ENCODER LAYER
class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff):
        super(EncoderLayer, self).__init__()
        self.FeedForwardNN = nn.Sequential(
            nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, dff)
        )

    def forward(self, x):
        output = self.FeedForwardNN(x)
        return output


# THE DECODER LAYER
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()
        self.MultiHAttention1 = MultiHeadAttention(d_model, num_heads)
        self.MultiHAttention2 = MultiHeadAttention(d_model, num_heads)
        self.FeedForwardNN = nn.Sequential(
            nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model)
        )
        self.layerNorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # print(f"FIRST MHA WITH LOOK AHEAD MASK")
        attn_output1 = self.MultiHAttention1(x, x, x, look_ahead_mask)
        attn_output1 = self.layerNorm1(x + attn_output1)
        # print(f"decoder input into second multihead attention layer:{attn_output1.shape}")
        attn_output2 = self.MultiHAttention2(
            enc_output, enc_output, attn_output1, padding_mask
        )
        attn_output2 = self.layerNorm2(attn_output2 + attn_output1)

        Feedforward_output = self.FeedForwardNN(attn_output2)
        final_output = self.layerNorm3(attn_output2 + Feedforward_output)
        return final_output


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            target_vocab_size, d_model
        )  # d_model is the size of embedding vector
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        )

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        return x
