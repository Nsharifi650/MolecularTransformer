import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

def create_padding_mask(seq):
    seq_masked = torch.tensor(seq) == 0 # True if value is 0 otherwise false
    return seq_masked.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size):
    # creating an upper triangle of 1s
    mask = torch.triu(torch.ones((size, size)), diagonal=1) 
    return mask.unsqueeze(0).unsqueeze(1)


import torch.optim as optim
import torch.nn as nn

def loss_function(real, pred):
    mask = real != 0
    loss = nn.CrossEntropyLoss(reduction='none')(pred.transpose(1, 2), real)
    mask = mask.float()
    loss *= mask

    return torch.mean(loss)

def train_model(transformer, train_loader, num_epochs, learning_rate, model_name, pretrained):

    if pretrained:
        transformer.load_state_dict(torch.load(model_name))
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0
        
        for idx_num, (properties, smiles) in enumerate(train_loader):
   
            properties = properties.to(device)
            smiles = smiles.to(device)
            look_ahead_mask = create_look_ahead_mask(smiles.size(1))
            dec_padding_mask = create_padding_mask(smiles)
            
            optimizer.zero_grad()
            predictions = transformer(properties, smiles, look_ahead_mask, dec_padding_mask, training=True)
            loss = loss_function(smiles[:, 1:], predictions[:, :-1])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print("batch loss", loss.item())
            # save model at the end of each epoch
            torch.save(transformer.state_dict(), model_name)
            
        print(f'Epoch {epoch+1}, Loss: {total_loss / (idx_num + 1)}')

# Initialize the model
target_vocab_size = len(char_to_idx)
print("target vocab size", target_vocab_size)
num_layers = 8
enc_d_model = 5 # number of properties
dec_d_model = 128
enc_num_heads = 1
dec_num_heads = 8
enc_dff = 128 # dimension of the feed forward layer
dec_dff = enc_dff
pe_target = 1000 # positional encoding
model_name = "molecularTransformer2.pth"
learning_rate = 1e-5
num_epochs = 20
pretrained = True

transformer = Transformer(num_layers, enc_d_model, dec_d_model,
                          enc_num_heads, dec_num_heads, enc_dff, 
                          dec_dff, target_vocab_size, pe_target)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = transformer.to(device)

# Train the model
train_model(transformer, train_loader, num_epochs, learning_rate, model_name, pretrained)
