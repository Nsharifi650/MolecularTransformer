from pydantic import BaseModel

class Model_config(BaseModel):
    num_layers: int
    enc_d_model: int
    dec_d_model: int
    dec_num_heads: int
    enc_dff: int 
    pe_target: int 
    model_name: str

class dataconfig(BaseModel):
    raw_csv_path: str
    processed_dataset_path: str
    test_data_size: float

class training_config(BaseModel):
    num_epochs: int
    pretrained: bool
    save_freq: int
    learning_rate: float
    batch_size: int

class Config(BaseModel):
    data: dataconfig
    model: Model_config
    training: training_config

