from dataclasses import dataclass
import yaml

@dataclass
class DataLoaderConfig:
  BatchSize: int =  32
  num_workers: int = 4
  shuffle: bool = True
  Transforms: bool = False
  Path: str = '/content/drive/MyDrive/Projects/data/nabirds/nabirds/images'
  deepLake: bool = True

@dataclass
class ModelConfig:
    Pretrained: bool = True
    Debug: bool = False
    Path: str = '/content/drive/MyDrive/Projects/ResNet.pt'

@dataclass
class TrainConfig:
    Epoch: int = 10
    WandB: bool = False

@dataclass
class OptimizerConfig:
    lr: float = 0.001
    momentum: float = 0.9

@dataclass
class SchedulerConfig:
    step_size: int = 7
    gamma: float = 0.1

@dataclass
class Config:
    DataLoader: DataLoaderConfig
    Model: ModelConfig
    Train: TrainConfig
    Optimizer: OptimizerConfig
    Scheduler: SchedulerConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DataLoader_config = DataLoaderConfig(**config_dict["DataLoader"])
        Model_config = ModelConfig(**config_dict["Model"])
        Train_config = TrainConfig(**config_dict["Train"])
        Optimizer_config = OptimizerConfig(**config_dict["Optimizer"])
        Scheduler_config = SchedulerConfig(**config_dict["Scheduler"])

        return Config(
            DataLoader=DataLoader_config,
            Model=Model_config,
            Train=Train_config,
            Optimizer=Optimizer_config,
            Scheduler=Scheduler_config
        )