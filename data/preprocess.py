import os

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.logging_config import get_logger

logger = get_logger(__name__)

def save_dataset(
        properties: list[float],
        smiles_indices: list[int],
        idx_to_char: dict[int, str],
        char_to_idx: dict[str, int],
        test_size: float,
        scaler: StandardScaler,
        output_dir: str
    ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_props, test_props, train_smiles, test_smiles = train_test_split(properties, smiles_indices, test_size=test_size)

    with open(os.path.join(output_dir, "train_dataset.pkl"), 'wb') as f:
        pickle.dump((train_props, train_smiles), f)
    
    with open(os.path.join(output_dir, "test_dataset.pkl"), 'wb') as f:
        pickle.dump((test_props, test_smiles), f)
    
    with open(os.path.join(output_dir, "char_mappings.pkl"), "wb") as f:
        pickle.dump((idx_to_char, char_to_idx), f)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    logger.info(f"Saved preprocessing datasets and mappings in the directory: {output_dir}")


def preprocess_data(
        csv_file: str
    ) -> tuple[list[float],list[int],dict[str,int],dict[int, str], StandardScaler]:
    data = pd.read_csv(csv_file)
    # removing any rows which have missing isosmiles
    data = data.dropna(subset=['isosmiles'])
    properties = data[['polararea', 'complexity', 'heavycnt', 'hbonddonor', 'hbondacc']].values
    smiles = data['isosmiles'].values
    logger.info(f"length of smiles: {smiles.shape}")
    # print(f"smiles: {smiles}")

    scaler = StandardScaler()
    properties = scaler.fit_transform(properties)
    
    # Convert SMILES to a list of character indices
    char_to_idx = {char: idx + 3 for idx, char in enumerate(sorted(set(''.join(smiles))))}
    char_to_idx['<pad>'] = 0
    char_to_idx['<start>'] = 1
    char_to_idx['<end>'] = 2

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    max_smiles_len = max(len(s) for s in smiles)+2 # 2 for the start and end token
    logger.info(f"max smiles length: {max_smiles_len}")
    smiles_indices = [
        [char_to_idx['<start>']] + [char_to_idx[char] for char in smi] + [char_to_idx['<end>']] + 
        [char_to_idx['<pad>']] * (max_smiles_len - len(smi) - 2)
        for smi in smiles
    ]
    logger.info("Data processing: conversion of SMILEs notation of molecules to sequence complete")

    return properties, smiles_indices, idx_to_char, char_to_idx,scaler


