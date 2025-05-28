import json

import lark_oapi
import optuna
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody, CreateMessageResponse
from optuna import Trial, samplers

from utils import Runner

best_f1 = 0


def objective(trial: Trial) -> float:
    global best_f1
    runner = Runner(trial)
    precision, recall, fpr, fnr, f1 = runner.run()
    if f1 > best_f1:
        best_f1 = f1

        client = (
            lark_oapi.Client.builder()
            .app_id('cli_a8bc4f1ba1ff100c')
            .app_secret('GSPcmc4qSw6t1CMbxOiUg2knwIuDJbzr')
            .build()
        )
        temp = {
            'text': '\n'.join(
                [
                    'New best result:',
                    f' - F1: {f1}',
                    f' - Precision: {precision}',
                    f' - Recall: {recall}',
                    f' - FPR: {fpr}',
                    f' - FNR: {fnr}'
                ]
            )
        }
        request: CreateMessageRequest = (
            CreateMessageRequest.builder()
            .receive_id_type('open_id')
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id('ou_f4463d383724ed844c1cd2b4c938c32d')
                .msg_type('text')
                .content(json.dumps(temp))
                .build()
            )
            .build()
        )

        response: CreateMessageResponse = client.im.v1.message.create(request)

    return f1


if __name__ == '__main__':
    db_string = f'sqlite:///../optuna.db'
    study = optuna.create_study(
        study_name='Hetero - New Network - 0528', direction='maximize', storage=db_string, load_if_exists=True, sampler=samplers.RandomSampler()
    )

    study.optimize(objective, n_trials=1000)

    print(study.best_params)
    print(study.best_trial)
