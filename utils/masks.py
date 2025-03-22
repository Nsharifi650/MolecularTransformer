import torch

def create_padding_mask(seq):
    seq_masked = torch.tensor(seq) == 0 # True if value is 0 otherwise false
    return seq_masked.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size: int):
    # creating an upper triangle of 1s
    mask = torch.triu(torch.ones((size, size)), diagonal=1) 
    return mask.unsqueeze(0).unsqueeze(1)