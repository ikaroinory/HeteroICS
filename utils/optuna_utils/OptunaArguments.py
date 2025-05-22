from typing import Literal

import torch
from optuna import Trial


class OptunaArguments:
    def __init__(self, trial: Trial):
        self.seed: int = 42

        self.model_path: str | None = None

        self.report: Literal['label', 'no_label'] = 'label'

        self.dataset: str = 'swat'
        self.dtype = torch.float32
        self.device = 'cuda'

        self.batch_size: int = trial.suggest_categorical('batch_size', [32, 64, 128])  # best: 256
        self.epochs: int = 100

        self.slide_window: int = trial.suggest_int('slide_window', 5, 50)
        self.slide_stride: int = 1
        # best: 5
        self.k_dict: dict[tuple[str, str, str], int] = {
            ('sensor', 'ss', 'sensor'): trial.suggest_int('k_ss', 1, 10),
            ('sensor', 'sa', 'actuator'): trial.suggest_int('k_sa', 1, 10),
            ('actuator', 'as', 'sensor'): trial.suggest_int('k_as', 1, 10),
            ('actuator', 'aa', 'actuator'): trial.suggest_int('k_aa', 1, 10)
        }

        self.d_hidden: int = trial.suggest_categorical('d_hidden', [64, 128, 256])  # best: 256
        self.d_output_hidden: int = trial.suggest_int('d_output_hidden', 128, 512, log=True)  # best: 128

        self.num_heads: int = trial.suggest_categorical('num_heads', [1, 2, 4, 8])  # best: 1
        self.num_output_layer: int = trial.suggest_int('num_output_layer', 1, 5)  # best: 2

        self.lr: float = trial.suggest_float('lr', 9e-5, 1e-2, log=True)  # best: 0.001
        self.decay: float = trial.suggest_float('lr', 9e-5, 1e-2, log=True)

        self.early_stop: int = 10

        self.log: bool = False
