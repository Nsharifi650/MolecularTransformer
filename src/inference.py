import os

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import pickle
import yaml

from src.config import Config
from src.model import Transformer
from utils.logging_config import get_logger
from utils.masks import create_look_ahead_mask


logger = get_logger(__name__)

def load_config(file_path: str) -> Config:
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def greedy_decode(
    model: Transformer,
    properties: torch.Tensor,
    max_length: int,
    start_token_idx: int,
    end_token_idx: int,
    device: torch.device
) -> list[int]:
    output_sequence = torch.tensor([[start_token_idx]], dtype=torch.long).to(device)

    for _ in range(max_length):
        look_ahead_mask = create_look_ahead_mask(output_sequence.size(1)).to(device)
        dec_padding_mask = None

        predictions = model(properties, output_sequence, look_ahead_mask, dec_padding_mask, training=False)
        predictions = predictions[:, -1:]

        output_sequence = torch.cat([output_sequence, predictions], dim=-1)
        if predictions.item() == end_token_idx:
            break

    return output_sequence.squeeze().tolist()

def preprocess_properties(
    csv_file: str, scaler_path: str
) -> np.ndarray:
    data = pd.read_csv(csv_file)
    properties = data[['polararea', 'complexity', 'heavycnt', 'hbonddonor', 'hbondacc']].values

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    properties = scaler.transform(properties)
    return properties

def run_inference(
    input_csv: str,
    output_csv: str
):
    config = load_config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mappings_path = os.path.join(config.data.processed_dataset_path, "char_mappings.pkl")
    with open(mappings_path, "rb") as f:
        idx_to_char, char_to_idx = pickle.load(f)

    start_token_idx = char_to_idx['<start>']
    end_token_idx = char_to_idx['<end>']

    scaler_path = os.path.join(config.data.processed_dataset_path, "scaler.pkl")
    properties = preprocess_properties(input_csv, scaler_path)


    model = Transformer(config, target_vocab_size=len(char_to_idx))
    model.load_state_dict(torch.load(config.model.model_name, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    for i in range(properties.shape[0]):
        prop_tensor = torch.tensor(properties[i], dtype=torch.float32).unsqueeze(0).to(device)
        generated_ids = greedy_decode(model, prop_tensor, max_length=288, start_token_idx=start_token_idx, end_token_idx=end_token_idx, device=device)
        smiles = ''.join([idx_to_char[idx] for idx in generated_ids if idx in idx_to_char and idx != 0])
        predictions.append(smiles)

    input_df = pd.read_csv(input_csv)
    input_df["predicted_smiles"] = predictions
    input_df.to_csv(output_csv, index=False)
    logger.info(f"Saved predictions to {output_csv}")
