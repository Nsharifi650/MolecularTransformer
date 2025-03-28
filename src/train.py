import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from src.model import Transformer
from src.dataset import load_dataset
from utils.logging_config import get_logger
from src.config import Config
from utils.masks import create_look_ahead_mask, create_padding_mask

logger = get_logger(__name__)


def loss_function(real, pred):
    mask = real != 0
    loss = nn.CrossEntropyLoss(reduction="none")(pred.transpose(1, 2), real)
    mask = mask.float()
    loss *= mask
    return torch.mean(loss)


def train_model(
    model: Transformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Config,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.training.pretrained:
        model.load_state_dict(torch.load(config.model.model_name))
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    model = model.to(device)

    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0

        for properties, smiles in train_loader:
            properties = properties.to(device)
            smiles = smiles.to(device)
            look_ahead_mask = create_look_ahead_mask(smiles.size(1))
            dec_padding_mask = create_padding_mask(smiles)

            optimizer.zero_grad()
            predictions = model(properties, smiles, look_ahead_mask, dec_padding_mask)
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
                predictions = model(
                    properties, smiles, look_ahead_mask, dec_padding_mask
                )
                loss = loss_function(smiles[:, 1:], predictions[:, :-1])
                test_error += loss.item()

        print(
            f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}, validation Loss: {test_error / len(test_loader)}"
        )
        if (epoch + 1) % config.training.save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join("saved_models", config.model.model_name),
            )


def run_training_pipeline(config: Config) -> None:
    train_loader, test_loader, char_to_idx, idx_to_char = load_dataset(
        config.data.processed_dataset_path, config.training.batch_size
    )

    # instantiate the model
    target_vocab_size = len(char_to_idx)

    transformer = Transformer(config, target_vocab_size)

    logger.info("Model training starting...")
    train_model(transformer, train_loader, test_loader, config)
    logger.info("Model training complete")
