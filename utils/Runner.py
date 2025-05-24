import copy
import json
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
from tqdm import tqdm

from datasets import HeteroICSDataset
from models import HeteroICS
from .Arguments import Arguments
from .Logger import Logger
from .OptunaArguments import OptunaArguments
from .evaluate import get_metrics


class Runner:
    def __init__(self, trail: Trial = None):
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
            node_indices=node_indices,
            edge_types=edge_types,
            dtype=self.__args.dtype,
            device=self.__args.device
        )
        self.__optimizer = Adam(self.__model.parameters(), lr=self.__args.lr)
        self.__loss = L1Loss()

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

        train_dataloader, valid_dataloader = self.__get_train_and_valid_dataloader(train_dataset, 0.1)

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

    def __train(self) -> float:
        Logger.info('Training...')

        best_epoch = -1
        best_train_loss = float('inf')
        best_model_weights = copy.deepcopy(self.__model.state_dict())
        no_improve_count = 0

        for epoch in tqdm(range(self.__args.epochs)):
            train_loss = self.__train_epoch()

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')

            if train_loss < best_train_loss:
                best_epoch = epoch + 1

                best_train_loss = train_loss

                best_model_weights = copy.deepcopy(self.__model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if no_improve_count >= self.__args.early_stop:
                break

        self.__model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_weights, self.__model_path)

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Train loss: {best_train_loss:.8f}')
        Logger.info(f'Model save to {self.__model_path}')

        return best_train_loss

    def __evaluate(self, model_name: Path) -> tuple[float, float, float, float]:
        Logger.info('Evaluating...')

        self.__model.load_state_dict(torch.load(f'{model_name}', weights_only=True))

        _, valid_result = self.__valid_epoch(self.__valid_dataloader)
        test_loss, test_result = self.__valid_epoch(self.__test_dataloader)

        Logger.info(f' - Test loss: {test_loss:.8f}')

        f1, precision, recall, auc = get_metrics(test_result, valid_result if self.__args.report == 'label' else None)

        Logger.info(f' - F1 score: {f1:.4f}')
        Logger.info(f' - Precision: {precision:.4f}')
        Logger.info(f' - Recall: {recall:.4f}')
        Logger.info(f' - AUC: {auc:.4f}')

        return f1, precision, recall, auc

    def run(self) -> tuple[float, float, float, float, float]:
        best_train_loss = self.__train()
        f1, precision, recall, auc = self.__evaluate(self.__model_path)
        return best_train_loss, f1, precision, recall, auc
