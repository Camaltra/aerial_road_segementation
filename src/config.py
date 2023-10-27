from dataclasses import dataclass
import numpy as np
import torch


@dataclass()
class ModelConfig:
    learning_rate: np.float16 = 1e-3
    batch_size: int = 16
    num_epoch: int = 100
    image_height: int = 256
    image_width: int = 256
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    def print_config(self):
        print("CONFIG TRAINING:")
        print(f"| Learning rate : {self.learning_rate}")
        print(f"| Batch size    : {self.batch_size}")
        print(f"| Num epoch     : {self.num_epoch}")
        print(f"| Image Height  : {self.image_height}")
        print(f"| Device        : {self.device}")
