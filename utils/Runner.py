import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from optuna import Trial
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import HeteroICSDataset
from models import HeteroICS
from .Arguments import Arguments
from .Logger import Logger
from .OptunaArguments import OptunaArguments
from .evaluate import get_metrics


class Runner:
    def __init__(self, trail: Trial = None):
        self.__writer = SummaryWriter()
        self.__args = OptunaArguments(trail) if trail is not None else Arguments()

        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.__log_path = Path(f'logs/{self.__args.dataset}/{self.start_time}.log')
        self.__model_path = Path(f'saves/{self.__args.dataset}/{self.start_time}.pth')

        Logger.init(self.__log_path if self.__args.log else None)

        Logger.info('Setting seed...')
        self.__set_seed()

        Logger.info('Loading data...')
        train_dataloader, valid_dataloader, test_dataloader = self.__get_dataloaders()

        self.__train_dataloader: DataLoader = train_dataloader
        self.__valid_dataloader: DataLoader = valid_dataloader
        self.__test_dataloader: DataLoader = test_dataloader

        with open(f'data/processed/{self.__args.dataset}/node_indices.json', 'r') as f:
            node_indices = json.load(f)
        with open(f'data/processed/{self.__args.dataset}/edge_types.json', 'r') as f:
            edge_types = json.load(f)
            edge_types: list[tuple[str, str, str]] = [tuple(edge_type) for edge_type in edge_types]

        Logger.info('Building model...')
        self.__model = HeteroICS(
            sequence_len=self.__args.slide_window,
            d_hidden=self.__args.d_hidden,
            d_output_hidden=self.__args.d_output_hidden,
            num_heads=self.__args.num_heads,
            num_output_layer=self.__args.num_output_layer,
            k_dict=self.__args.k_dict,
            dropout=self.__args.dropout,
            node_indices=node_indices,
            edge_types=edge_types,
            dtype=self.__args.dtype,
            device=self.__args.device
        )
        self.__loss = L1Loss()
        self.__optimizer = Adam(self.__model.parameters(), lr=self.__args.lr)

    def __set_seed(self) -> None:
        os.environ['PYTHONHASHSEED'] = str(self.__args.seed)

        random.seed(self.__args.seed)

        np.random.seed(self.__args.seed)

        torch.manual_seed(self.__args.seed)
        torch.cuda.manual_seed(self.__args.seed)
        torch.cuda.manual_seed_all(self.__args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __get_train_and_valid_dataloader(self, train_dataset: HeteroICSDataset, valid_size: float) -> tuple[DataLoader, DataLoader]:
        dataset_size = int(len(train_dataset))
        train_dataset_size = int((1 - valid_size) * dataset_size)
        valid_dataset_size = int(valid_size * dataset_size)

        valid_start_index = random.randrange(train_dataset_size)

        indices = torch.arange(dataset_size)
        train_indices = torch.cat([indices[:valid_start_index], indices[valid_start_index + valid_dataset_size:]])
        valid_indices = indices[valid_start_index:valid_start_index + valid_dataset_size]

        train_subset = Subset(train_dataset, train_indices)
        valid_subset = Subset(train_dataset, valid_indices)

        train_dataloader = DataLoader(train_subset, batch_size=self.__args.batch_size, shuffle=True, worker_init_fn=lambda _: self.__set_seed())

        valid_dataloader = DataLoader(valid_subset, batch_size=self.__args.batch_size, shuffle=False, worker_init_fn=lambda _: self.__set_seed())

        return train_dataloader, valid_dataloader

    def __get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_df = pd.read_csv(f'data/processed/{self.__args.dataset}/train.csv')
        train_np = train_df.to_numpy()
        test_df = pd.read_csv(f'data/processed/{self.__args.dataset}/test.csv')
        test_np = test_df.to_numpy()

        train_dataset = HeteroICSDataset(
            train_np,
            self.__args.slide_window,
            self.__args.slide_stride,
            mode='train',
            dtype=self.__args.dtype
        )
        test_dataset = HeteroICSDataset(
            test_np,
            self.__args.slide_window,
            self.__args.slide_stride,
            mode='test',
            dtype=self.__args.dtype
        )

        train_dataloader, valid_dataloader = self.__get_train_and_valid_dataloader(train_dataset, 0.1)

        test_dataloader = DataLoader(test_dataset, batch_size=self.__args.batch_size, shuffle=False, worker_init_fn=lambda _: self.__set_seed())

        return train_dataloader, valid_dataloader, test_dataloader

    def __train_epoch(self) -> float:
        self.__model.train()

        total_train_loss = 0
        for x, y, _ in tqdm(self.__train_dataloader):
            x = x.to(self.__args.device)
            y = y.to(self.__args.device)

            self.__optimizer.zero_grad()

            output = self.__model(x)

            loss = self.__loss(output, y)

            loss.backward()
            self.__optimizer.step()

            total_train_loss += loss.item() * x.shape[0]

        return total_train_loss / len(self.__train_dataloader.dataset)

    def __valid_epoch(self, dataloader: DataLoader) -> tuple[float, tuple[Tensor, Tensor, Tensor]]:
        self.__model.eval()

        predicted_list = []
        actual_list = []
        label_list = []

        total_valid_loss = 0
        for x, y, label in tqdm(dataloader):
            x = x.to(self.__args.device)
            y = y.to(self.__args.device)
            label = label.to(self.__args.device)

            with torch.no_grad():
                output = self.__model(x)

                loss = self.__loss(output, y)

                total_valid_loss += loss.item() * x.shape[0]

                predicted_list.append(output)
                actual_list.append(y)
                label_list.append(label)

        predicted_tensor = torch.cat(predicted_list, dim=0)
        actual_tensor = torch.cat(actual_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return total_valid_loss / len(dataloader.dataset), (predicted_tensor, actual_tensor, label_tensor)

    def __train(self) -> None:
        Logger.info('Training...')

        best_epoch = -1
        best_valid_loss = float('inf')
        best_model_weights = copy.deepcopy(self.__model.state_dict())
        patience_counter = 0

        for epoch in tqdm(range(self.__args.epochs)):
            train_loss = self.__train_epoch()
            valid_loss, _ = self.__valid_epoch(self.__valid_dataloader)

            self.__writer.add_scalar('Loss/train', train_loss, epoch)
            self.__writer.add_scalar('Loss/valid', valid_loss, epoch)

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')
            Logger.info(f' - Valid loss: {valid_loss:.8f}')

            if valid_loss < best_valid_loss:
                best_epoch = epoch + 1

                best_valid_loss = valid_loss

                best_model_weights = copy.deepcopy(self.__model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if patience_counter >= self.__args.early_stop:
                break

        self.__model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_weights, self.__model_path)

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Valid loss: {best_valid_loss:.8f}')
        Logger.info(f'Model save to {self.__model_path}')

    def __evaluate(self, model_name: Path) -> tuple[float, float, float, float, float]:
        Logger.info('Evaluating...')

        self.__model.load_state_dict(torch.load(f'{model_name}', weights_only=True))

        _, test_result = self.__valid_epoch(self.__test_dataloader)
        if self.__args.report == 'label':
            _, valid_result = self.__valid_epoch(self.__valid_dataloader)
            precision, recall, fpr, fnr, f1 = get_metrics(test_result, valid_result)
        else:
            precision, recall, fpr, fnr, f1 = get_metrics(test_result)

        Logger.info(f' - F1 score: {f1:.4f}')
        Logger.info(f' - Precision: {precision:.4f}')
        Logger.info(f' - Recall: {recall:.4f}')
        Logger.info(f' - FPR: {fpr:.4f}')
        Logger.info(f' - FNR: {fnr:.4f}')

        return precision, recall, fpr, fnr, f1

    def run(self) -> tuple[float, float, float, float, float]:
        if self.__args.model_path is None:
            self.__train()
            precision, recall, fpr, fnr, f1 = self.__evaluate(self.__model_path)
        else:
            precision, recall, fpr, fnr, f1 = self.__evaluate(Path(self.__args.model_path))

        self.__writer.close()
        return precision, recall, fpr, fnr, f1
