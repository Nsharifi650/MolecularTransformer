import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from src.model import Transformer
from src.dataset import load_dataset
from utils.logging_config import get_logger

logger = get_logger(__name__)

def create_padding_mask(seq):
    seq_masked = torch.tensor(seq) == 0 # True if value is 0 otherwise false
    return seq_masked.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size):
    # creating an upper triangle of 1s
    mask = torch.triu(torch.ones((size, size)), diagonal=1) 
    return mask.unsqueeze(0).unsqueeze(1)

def loss_function(real, pred):
    mask = real != 0
    loss = nn.CrossEntropyLoss(reduction='none')(pred.transpose(1, 2), real)
    mask = mask.float()
    loss *= mask
    return torch.mean(loss)

def train_model(
    model: Transformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    model_name: str,
    pretrained: bool,
    save_freq: int = 2,
    ) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if pretrained:
        model.load_state_dict(torch.load(model_name))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for properties, smiles in train_loader:
            properties = properties.to(device)
            smiles = smiles.to(device)
            look_ahead_mask = create_look_ahead_mask(smiles.size(1))
            dec_padding_mask = create_padding_mask(smiles)
            
            optimizer.zero_grad()
            predictions = model(properties, smiles, look_ahead_mask, dec_padding_mask, training=True)
            loss = loss_function(smiles[:, 1:], predictions[:, :-1])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        with torch.no_grad():
            test_error = 0
            model.eval()
            for properties, smiles in test_loader:
                properties = properties.to(device)
                smiles = smiles.to(device)
                look_ahead_mask = create_look_ahead_mask(smiles.size(1))
                dec_padding_mask = create_padding_mask(smiles)
                predictions = model(properties, smiles, look_ahead_mask, dec_padding_mask, training=True)
                loss = loss_function(smiles[:, 1:], predictions[:, :-1])
                test_error += loss.item()

        print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}, validation Loss: {test_error/ len(test_loader)}')
        if (epoch+1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join("saved_models", model_name))

def run_training_pipeline(
        dataset_dir: str,
        batch_size: int,
        config
        ) -> None:
    train_loader, test_loader, char_to_idx, idx_to_char = load_dataset(
        dataset_dir, batch_size)
    
    # instantiate the model
    target_vocab_size = len(char_to_idx)
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
    save_freq = 2
    transformer = Transformer(num_layers, enc_d_model, dec_d_model,
                            enc_num_heads, dec_num_heads, enc_dff, 
                            dec_dff, target_vocab_size, pe_target)
    
    logger.info("Model training starting...")
    train_model(transformer, train_loader, test_loader,
                num_epochs, learning_rate, model_name, 
                pretrained, save_freq
            )
    logger.info("Model training complete")
