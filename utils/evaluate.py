import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor


def get_total_error_score(result: tuple[Tensor, Tensor, Tensor], smooth_window: int, epsilon: float = 1e-2) -> Tensor:
    predict_result, actual_result, _ = result

    delta = torch.abs(predict_result - actual_result)
    median, _ = torch.median(delta, dim=0)
    iqr = torch.quantile(delta, 0.75, dim=0) - torch.quantile(delta, 0.25, dim=0)
    score = (delta - median) / (torch.abs(iqr) + epsilon)  # [*, num_nodes]

    if smooth_window > 0:
        score = score.unfold(0, smooth_window, 1).mean(-1)
        score = F.pad(score, (0, 0, smooth_window - 1, 0), mode='constant')

    return score.max(dim=-1)[0]


def get_f1_with_label(error_score: Tensor, actual_label: Tensor) -> tuple[float, Tensor]:
    steps = 400

    threshold_rank_list = torch.tensor(np.linspace(0, error_score.shape[-1], steps, endpoint=False))
    _, index = error_score.sort(descending=True)
    rank = (error_score.argsort() + 1).argsort() + 1

    f1_list = []
    predict_label_list = []
    for threshold_rank in threshold_rank_list:
        predict_label = (rank > threshold_rank).long().cpu()

        f1_list.append(f1_score(actual_label, predict_label))
        predict_label_list.append(predict_label)

    f1_list = torch.tensor(f1_list)

    _, indices = torch.topk(f1_list, 1)

    index = indices[0]

    return f1_list[index].item(), predict_label_list[index]


def get_metrics(
    test_result: tuple[Tensor, Tensor, Tensor],
    valid_result: tuple[Tensor, Tensor, Tensor] = None,
    smooth_window: int = 4
) -> tuple[float, float, float, float]:
    test_error_score = get_total_error_score(test_result, smooth_window)

    actual_labels = test_result[2].cpu()
    if valid_result is not None:
        valid_error_score = get_total_error_score(valid_result, smooth_window)

        threshold = torch.max(valid_error_score)

        predict_labels = (test_error_score > threshold).long().cpu()

        f1 = f1_score(actual_labels, predict_labels)
    else:
        f1, predict_labels = get_f1_with_label(test_error_score, actual_labels)

    precision = precision_score(actual_labels, predict_labels)
    recall = recall_score(actual_labels, predict_labels)
    auc = roc_auc_score(actual_labels, predict_labels)

    return f1, precision, recall, auc
