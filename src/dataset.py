import os

import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, properties, smiles):
        self.properties = properties
        self.smiles = smiles

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        return torch.tensor(self.properties[idx], dtype=torch.float32), torch.tensor(
            self.smiles[idx], dtype=torch.long
        )


def load_dataset(
    data_dir: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The folder {data_dir} does not exist")

    with open(os.path.join(data_dir, "train_dataset.pkl"), "rb") as f:
        train_props, train_smiles = pickle.load(f)

    with open(os.path.join(data_dir, "test_dataset.pkl"), "rb") as f:
        test_props, test_smiles = pickle.load(f)

    with open(os.path.join(data_dir, "char_mappings.pkl"), "rb") as f:
        idx_to_char, char_to_idx = pickle.load(f)

    train_dataset = MoleculeDataset(train_props, train_smiles)
    test_dataset = MoleculeDataset(test_props, test_smiles)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, char_to_idx, idx_to_char
