import argparse
from typing import Literal

import torch


class Arguments:
    def __init__(self):
        args = self.parse_args()

        self.seed: int = args.seed

        self.model_path: str | None = args.model

        self.report: Literal['label', 'no_label'] = args.report

        self.dataset: str = args.dataset
        self.dtype = torch.float32 if args.dtype == 'float32' or args.dtype == 'float' else torch.float64
        self.device = args.device

        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs

        self.slide_window: int = args.slide_window
        self.slide_stride: int = args.slide_stride
        self.k_dict: dict[tuple[str, str, str], int] = {
            ('sensor', 'ss', 'sensor'): args.k[0],
            ('sensor', 'sa', 'actuator'): args.k[1],
            ('actuator', 'as', 'sensor'): args.k[2],
            ('actuator', 'aa', 'actuator'): args.k[3]
        }

        self.d_hidden: int = args.d_hidden
        self.d_output_hidden: int = args.d_output_hidden

        self.num_heads: int = args.num_heads
        self.num_output_layer: int = args.num_output_layer

        self.lr: float = args.lr
        self.dropout: float = args.dropout

        self.early_stop: int = args.early_stop

        self.log: bool = not args.nolog

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--model', type=str)

        parser.add_argument('--report', type=str, choices=['label', 'no_label'], default='label')

        parser.add_argument('-ds', '--dataset', type=str, default='swat')
        parser.add_argument('--dtype', choices=['float', 'double'], default='float')
        parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

        parser.add_argument('-b', '--batch_size', type=int, default=64)
        parser.add_argument('-e', '--epochs', type=int, default=100)

        parser.add_argument('-sw', '--slide_window', type=int, default=9)
        parser.add_argument('-ss', '--slide_stride', type=int, default=1)
        parser.add_argument('-k', '--k', type=int, nargs='*')

        parser.add_argument('--d_hidden', type=int, default=64)
        parser.add_argument('--d_output_hidden', type=int, default=132)

        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_output_layer', type=int, default=4)

        parser.add_argument('--lr', type=float, default=0.002)
        parser.add_argument('--dropout', type=float, default=0.15)

        parser.add_argument('--early_stop', type=int, default=10)

        parser.add_argument('--nolog', action='store_true')

        return parser.parse_args()
