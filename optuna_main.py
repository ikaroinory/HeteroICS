import optuna
from optuna import Trial

from utils.optuna_utils import OptunaRunner


def objective(trial: Trial) -> tuple[float, float, float]:
    f1_list = []
    for _ in range(5):
        runner = OptunaRunner(trial)
        best_train_loss_with_best_epoch, best_valid_loss, f1, precision, recall, auc = runner.run()
        f1_list.append(f1)
    max_f1 = max(f1_list)
    min_f1 = min(f1_list)
    mean_f1 = sum(f1_list) / 5
    return max_f1, min_f1, mean_f1


if __name__ == '__main__':
    db_string = f'sqlite:///optuna.db'
    study = optuna.create_study(study_name='Hetero', directions=['maximize', 'maximize', 'maximize'], storage=db_string, load_if_exists=True)

    study.optimize(objective, n_trials=1000)

    print(study.best_params)
    print(study.best_trial)
