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
    length: int = 7
    sampling_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    downsample: bool = True
    mel_spectrogram = None
    verbose: bool = False
    fixed_limit: bool = True
    batch_size: int = 8
    batchnorm: bool = False
    num_workers: int = 1

    # Model args
    weight_decay: float = 0.0001
    learning_rate: float = 5e-4
    min_lr: float = 5e-9
    epochs: int = 35
    model_depth: int = 1
    data_mode = 'denoise'
    scheduler: Any = field(default=None)
    criterion: Any = field(default=None)
    optimizer: Any = field(default=None)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


    # save / load model
    save_path: str = './src/checkpoints/DGL.pth.tar'
    load_path: str = './src/checkpoints/DGL.pth.tar'

    def update_hyperparameter(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


hp = Hyperparameter()

