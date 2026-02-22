
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DataConfig:
    data_root: str
    images_path: str
    indices_path: str
    split: str = "train"
    subject: str = "subj01"
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 224

@dataclass
class ModelConfig:
    latent_dim: int = 1024
    hidden_dim: int = 2048
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "results"
    
@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            
        data_cfg = DataConfig(**cfg_dict.get('data', {}))
        model_cfg = ModelConfig(**cfg_dict.get('model', {}))
        training_cfg = TrainingConfig(**cfg_dict.get('training', {}))
        
        return cls(data=data_cfg, model=model_cfg, training=training_cfg)

def load_config(config_path: str) -> Config:
    return Config.from_yaml(config_path)
