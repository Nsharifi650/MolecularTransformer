import yaml

from data.preprocess import preprocess_data, save_dataset
from src.train import run_training_pipeline
from src.config import Config


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)


def main():
    configuration = load_config("config/config.yaml")
    properties, smiles_indices, idx_to_char, char_to_idx, scaler = preprocess_data(
        configuration.data.raw_csv_path
    )

    save_dataset(
        properties,
        smiles_indices,
        idx_to_char,
        char_to_idx,
        configuration.data.test_data_size,
        scaler,
        configuration.data.processed_dataset_path,
    )

    run_training_pipeline(configuration)


if __name__ == "__main__":
    main()
