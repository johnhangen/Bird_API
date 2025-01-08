from dataclasses import dataclass
import yaml

@dataclass
class DataLoaderConfig:
  BatchSize: int =  32
  num_workers: int = 4
  shuffle: bool = True
  Transforms: bool = False
  Path: str = r'data\nabirds\nabirds'

@dataclass
class ModelConfig:
    Pretrained: bool = True

@dataclass
class TrainConfig:
    Epoch: int = 10

@dataclass
class OptimizerConfig:
    lr: float = 0.001
    momentum: float = 0.9

@dataclass
class Config:
    DataLoader: DataLoaderConfig
    Model: ModelConfig
    Train: TrainConfig
    Optimizer: OptimizerConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DataLoader_config = DataLoaderConfig(**config_dict["DataLoader"])
        Model_config = ModelConfig(**config_dict["Model"])
        Train_config = TrainConfig(**config_dict["Train"])
        Optimizer_config = OptimizerConfig(**config_dict["Optimizer"])

        return Config(
            DataLoader=DataLoader_config,
            Model=Model_config,
            Train=Train_config,
            Optimizer=Optimizer_config
        )