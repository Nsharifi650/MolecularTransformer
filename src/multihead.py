# MULTIHEAD ATTENTION 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        try:
            assert d_model % num_heads == 0
        except Exception as e:
            logger.error(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_models = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # The query, key, value learnable matrices
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.FCLayer = nn.Linear(d_model, d_model)
    def split_embedding_perHead(self,x):

        (batch_size, seq_len, d_model) = x.shape
        x = x.view(batch_size, -1, self.num_heads, self.depth)

        x = x.permute(0,2,1,3)
        return x
    
    def cal_attention(self,q,k,v,mask):
        qk = torch.matmul(q, k.permute(0,1,3,2))
        dk=torch.tensor(k.shape[-1], dtype=torch.float32)
        #dk is a tensor scalar!
        attention = qk/torch.sqrt(dk)

        if mask is not None:
            attention += (mask*-1e9)

        attention_weights = F.softmax(attention, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
    
    def forward(self, v,k,q,mask):
        batch_size = q.shape[0]

        q = self.split_embedding_perHead(self.Wq(q))
        k = self.split_embedding_perHead(self.Wk(k))
        v = self.split_embedding_perHead(self.Wv(v))

        attention,atten_weights = self.cal_attention(q,k,v,mask)
        attention = attention.permute(0,2,1,3).contiguous()
        attention = attention.reshape(batch_size, -1, self.d_models)

        output = self.FCLayer(attention)
        return output
