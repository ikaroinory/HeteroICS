import torch
from torch import Tensor, nn
from torch.nn import functional as F

from enums import NodeConfig


class OutputLayer(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, num_layers: int, *, node_config: dict[str, NodeConfig], dtype=None, device=None):
        super().__init__()

        self.node_config = node_config

        self.num_layers = num_layers

        self.mlp = nn.ModuleList()
        self.output_linear_dict = nn.ModuleDict({
            node_type: nn.Linear(
                d_input if num_layers == 1 else d_hidden,
                1 if node_config['value_type'] == 'float' else len(node_config['value_list'])
            )
            for node_type, node_config in node_config.items()
        })
        for i in range(num_layers - 1):
            self.mlp.append(nn.Linear(d_input if i == 0 else d_hidden, d_hidden))
            self.mlp.append(nn.BatchNorm1d(d_hidden))
            self.mlp.append(nn.ReLU())

        self.dtype = dtype
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, d_input = x.shape

        mlp_output = x.reshape(-1, d_input)

        for model in self.mlp:
            mlp_output = model(mlp_output)
        mlp_output = mlp_output.reshape(batch_size, num_nodes, -1)

        output = torch.zeros([batch_size, num_nodes, 1], dtype=self.dtype, device=self.device)
        for node_type, output_linear in self.output_linear_dict.items():
            indices = torch.tensor(self.node_config[node_type]['index'], device=self.device)

            temp_output = output_linear(mlp_output[:, indices, :])
            temp_output = F.softmax(temp_output, dim=-1)

            output[:, indices, :] = temp_output.argmax(dim=-1, keepdim=True).to(self.dtype)

        return output

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
