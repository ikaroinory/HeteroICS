from typing import Literal

import torch
from optuna import Trial


class OptunaArguments:
    def __init__(self, trial: Trial):
        self.seed: int = 42

        self.model_path: str | None = None

        self.report: Literal['label', 'no_label'] = 'no_label'

        self.dataset: str = 'swat'
        self.dtype = torch.float32
        self.device = 'cuda'

        self.batch_size: int = 128
        self.epochs: int = 50

        self.slide_window: int = trial.suggest_int('slide_window', 5, 20)
        self.slide_stride: int = 1
        # best: 5
        self.k_dict: dict[tuple[str, str, str], int] = {
            ('sensor', 'ss', 'sensor'): trial.suggest_int('k_ss', 1, 10),
            ('sensor', 'sa', 'actuator'): trial.suggest_int('k_sa', 1, 10),
            ('actuator', 'as', 'sensor'): trial.suggest_int('k_as', 1, 10),
            ('actuator', 'aa', 'actuator'): trial.suggest_int('k_aa', 1, 10),
        }

        self.d_hidden: int = trial.suggest_categorical('d_hidden', [64, 128, 256])
        self.d_output_hidden: int = trial.suggest_int('d_output_hidden', 128, 256, log=True)

        self.num_heads: int = 8
        self.num_output_layer: int = trial.suggest_int('num_output_layer', 1, 10)  # best: 2

        self.share_lr: float = trial.suggest_float('share_lr', 1e-4, 1e-2, log=True)
        self.sensor_lr: float = trial.suggest_float('sensor_lr', 1e-4, 1e-2, log=True)
        self.actuator_lr: float = trial.suggest_float('actuator_lr', 1e-4, 1e-2, log=True)
        self.dropout: float = 0

        self.early_stop: int = 50

        self.log: bool = True
