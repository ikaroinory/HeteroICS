import json
import random

import numpy as np
import pandas as pd
import torch
from optuna import Trial
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import HeteroICSDataset
from models import HeteroICS
from utils import Logger
from utils.evaluate import get_metrics
from utils.optuna_utils import OptunaArguments


class OptunaRunner:
    def __init__(self, trail: Trial):
        self.__args = OptunaArguments(trail)

        Logger.init()

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
        self.__optimizer = Adam(self.__model.parameters(), lr=self.__args.lr)
        self.__loss = MSELoss()

    def __set_seed(self) -> None:
        random.seed(self.__args.seed)
        np.random.seed(self.__args.seed)
        torch.manual_seed(self.__args.seed)
        torch.cuda.manual_seed(self.__args.seed)

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

        self.__set_seed()
        train_dataloader = DataLoader(train_subset, batch_size=self.__args.batch_size, shuffle=True)

        self.__set_seed()
        valid_dataloader = DataLoader(valid_subset, batch_size=self.__args.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader

    def __get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_df = pd.read_csv(f'data/processed/{self.__args.dataset}/train.csv')
        train_np = train_df.to_numpy()
        test_np = pd.read_csv(f'data/processed/{self.__args.dataset}/test.csv').to_numpy()

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

        train_dataloader, valid_dataloader = self.__get_train_and_valid_dataloader(train_dataset, 0.2)

        self.__set_seed()
        test_dataloader = DataLoader(test_dataset, batch_size=self.__args.batch_size, shuffle=False)

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

            total_train_loss += loss.item()

        return total_train_loss / len(self.__train_dataloader)

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

                total_valid_loss += loss.item()

                predicted_list.append(output)
                actual_list.append(y)
                label_list.append(label)

        predicted_tensor = torch.cat(predicted_list, dim=0)
        actual_tensor = torch.cat(actual_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return total_valid_loss / len(self.__valid_dataloader), (predicted_tensor, actual_tensor, label_tensor)

    def __train(self) -> tuple[float, float]:
        Logger.info('Training...')

        best_epoch = -1
        best_train_loss_with_best_epoch = float('inf')
        best_valid_loss = float('inf')
        no_improve_count = 0

        for epoch in tqdm(range(self.__args.epochs)):
            train_loss = self.__train_epoch()
            valid_loss, _ = self.__valid_epoch(self.__valid_dataloader)

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')
            Logger.info(f' - Valid loss: {valid_loss:.8f}')

            if valid_loss < best_valid_loss:
                best_epoch = epoch + 1

                best_train_loss_with_best_epoch = train_loss
                best_valid_loss = valid_loss

                no_improve_count = 0
            else:
                no_improve_count += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if no_improve_count >= self.__args.early_stop:
                break

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Train loss: {best_train_loss_with_best_epoch:.8f}')
        Logger.info(f' - Valid loss: {best_valid_loss:.8f}')

        return best_train_loss_with_best_epoch, best_valid_loss

    def __evaluate(self) -> tuple[float, float, float, float]:
        Logger.info('Evaluating...')

        _, valid_result = self.__valid_epoch(self.__valid_dataloader)
        test_loss, test_result = self.__valid_epoch(self.__test_dataloader)

        Logger.info(f' - Test loss: {test_loss:.8f}')

        f1, precision, recall, auc = get_metrics(test_result, valid_result if self.__args.report == 'label' else None, self.__args.slide_window)

        Logger.info(f' - F1 score: {f1:.4f}')
        Logger.info(f' - Precision: {precision:.4f}')
        Logger.info(f' - Recall: {recall:.4f}')
        Logger.info(f' - AUC: {auc:.4f}')

        return f1, precision, recall, auc

    def run(self) -> tuple[float, float, float, float, float, float]:
        best_train_loss_with_best_epoch, best_valid_loss = self.__train()
        f1, precision, recall, auc = self.__evaluate()
        return best_train_loss_with_best_epoch, best_valid_loss, f1, precision, recall, auc
