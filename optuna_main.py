import optuna
from optuna import Trial

from utils import Runner


def objective(trial: Trial) -> float:
    runner = Runner(trial)
    precision, recall, fpr, fnr, f1 = runner.run()
    return f1


if __name__ == '__main__':
    db_string = f'sqlite:///optuna.db'
    study = optuna.create_study(study_name='Hetero', direction='maximize', storage=db_string, load_if_exists=True)

    study.optimize(objective, n_trials=1000)

    print(study.best_params)
    print(study.best_trial)
