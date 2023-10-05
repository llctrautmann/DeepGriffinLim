from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import torch


@dataclass
class Hyperparameter:
    # dataloader args
    annotation_file_path: str = './data/AvianID_AcousticIndices/EC_AI_cleaned.csv'
    root_dir: str = './data/EC_BIRD/'
    key: str = 'habitat'
    mode: str = 'stft'
    length: int = 5
    sampling_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    downsample: bool = False
    mel_spectrogram = None
    verbose: bool = False
    fixed_limit: bool = True
    batch_size: int = 16
    batchnorm: bool = False
    num_workers: int = 8

    # Model args
    subset_size: int = 800
    weight_decay: float = 0.0001
    learning_rate: float = 4e-6
    min_lr: float = 4e-10
    epochs: int = 150
    model_depth: int = 3
    data_mode = 'random'
    scheduler: Any = field(default=None)
    criterion: Any = field(default=None)
    optimizer: Any = field(default=None)
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


    # save / load model
    save_path: str = './src/checkpoints/DGL.pth.tar'
    load_path: str = './src/checkpoints/DGL.pth.tar'
    testing: bool = False

    def update_hyperparameter(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


hp = Hyperparameter()

