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
        card_cfg = {
            'schema': '2.0',
            'config': {
                'update_multi': True,
                'style': {'text_size': {'normal_v2': {'default': 'normal', 'pc': 'normal', 'mobile': 'heading'}}}
            },
            'body': {
                'direction': 'vertical',
                'horizontal_spacing': '8px',
                'vertical_spacing': '20px',
                'horizontal_align': 'left',
                'vertical_align': 'top',
                'padding': '20px 20px 20px 20px',
                'elements': [
                    {
                        'tag': 'markdown',
                        'content': '## :PARTY: Best Trial :PARTY:',
                        'text_align': 'center',
                        'text_size': 'normal_v2',
                        'margin': '0px 0px 0px 0px'
                    },
                    {
                        'tag': 'table',
                        'columns': [
                            {
                                'data_type': 'text',
                                'name': 'metric',
                                'display_name': 'Metric',
                                'horizontal_align': 'left',
                                'vertical_align': 'center',
                                'width': 'auto'
                            },
                            {
                                'data_type': 'number',
                                'name': 'value',
                                'display_name': 'Value',
                                'horizontal_align': 'right',
                                'vertical_align': 'center',
                                'width': 'auto',
                                'format': {'precision': 2}
                            }
                        ],
                        'rows': [
                            {'metric': 'F1 score', 'value': f1, },
                            {'metric': 'Precision', 'value': precision},
                            {'metric': 'Recall', 'value': recall},
                            {'metric': 'FPR', 'value': fpr},
                            {'metric': 'FNR', 'value': fnr}
                        ],
                        'row_height': 'low',
                        'header_style': {
                            'background_style': 'grey',
                            'bold': True,
                            'lines': 1
                        },
                        'page_size': 5,
                        'margin': '0px 0px 0px 0px'
                    },
                    {
                        'tag': 'table',
                        'columns': [
                            {
                                'data_type': 'text',
                                'name': 'parameter',
                                'display_name': 'Parameter',
                                'horizontal_align': 'left',
                                'vertical_align': 'center',
                                'width': 'auto'
                            },
                            {
                                'data_type': 'number',
                                'name': 'value',
                                'display_name': 'Value',
                                'horizontal_align': 'right',
                                'vertical_align': 'center',
                                'width': 'auto',
                                'format': {
                                    'precision': 6
                                }
                            }
                        ],
                        'rows': [
                            {'parameter': 'slide_window', 'value': runner.args.slide_window},
                            {'parameter': 'k_ss', 'value': runner.args.k[('sensor', 'ss', 'sensor')]},
                            {'parameter': 'k_sa', 'value': runner.args.k[('sensor', 'sa', 'actuator')]},
                            {'parameter': 'k_as', 'value': runner.args.k[('actuator', 'as', 'sensor')]},
                            {'parameter': 'k_aa', 'value': runner.args.k[('actuator', 'aa', 'actuator')]},
                            {'parameter': 'd_hidden', 'value': runner.args.d_hidden},
                            {'parameter': 'd_output_hidden', 'value': runner.args.d_output_hidden},
                            {'parameter': 'num_heads', 'value': runner.args.num_heads},
                            {'parameter': 'num_output_layer', 'value': runner.args.num_output_layer},
                            {'parameter': 'lr', 'value': runner.args.lr},
                            {'parameter': 'dropout', 'value': runner.args.dropout},
                        ],
                        'row_height': 'low',
                        'header_style': {
                            'background_style': 'grey',
                            'bold': True,
                            'lines': 1
                        },
                        'page_size': 5,
                        'margin': '0px 0px 0px 0px'
                    }
                ]
            }
        }
        request: CreateMessageRequest = (
            CreateMessageRequest.builder()
            .receive_id_type('open_id')
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id('ou_f4463d383724ed844c1cd2b4c938c32d')
                .msg_type('interactive')
                .content(json.dumps(card_cfg))
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
