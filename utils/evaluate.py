import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from torch import Tensor


def get_total_error_score(result: tuple[Tensor, Tensor, Tensor], smooth_window: int, epsilon: float = 1e-2) -> Tensor:
    predict_result, actual_result, _ = result

    delta = (predict_result - actual_result).abs()
    iqr = delta.quantile(0.75, dim=0) - delta.quantile(0.25, dim=0)
    score = (delta - delta.quantile(0.5, dim=0)) / (iqr + epsilon)  # [*, num_nodes]

    if smooth_window > 0:
        score = score.unfold(0, smooth_window, 1).mean(-1)
        score = F.pad(score, (0, 0, smooth_window - 1, 0), mode='constant')

    return score.max(dim=-1)[0]


def get_f1_with_label(error_score: Tensor, actual_label: Tensor) -> tuple[float, Tensor]:
    steps = 1000

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
    smooth_window: int = 0
) -> tuple[float, float, float, float, float]:
    test_error_score = get_total_error_score(test_result, smooth_window)

    actual_labels = test_result[2].cpu()
    if valid_result is not None:
        valid_error_score = get_total_error_score(valid_result, smooth_window)

        threshold = torch.max(valid_error_score)

        predict_labels = (test_error_score > threshold).long().cpu()

        f1 = f1_score(actual_labels, predict_labels)
    else:
        test_error_score = test_error_score.cpu()

        precision, recall, thresholds = precision_recall_curve(actual_labels, test_error_score)
        f1_score_list = 2 * precision * recall / (precision + recall + 1e-8)

        best_index = f1_score_list.argmax()
        best_threshold = thresholds[best_index]

        predict_labels = (test_error_score > best_threshold)
        f1 = f1_score_list[best_index]

    tn, fp, fn, tp = confusion_matrix(actual_labels, predict_labels).ravel()

    precision = precision_score(actual_labels, predict_labels)
    recall = recall_score(actual_labels, predict_labels)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    return precision, recall, fpr, fnr, f1
