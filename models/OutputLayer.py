from torch import Tensor, nn


class OutputLayer(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, num_layers: int):
        super().__init__()

        self.num_layers = num_layers

        self.mlp = nn.ModuleList()
        if num_layers == 1:
            self.mlp.append(nn.Linear(d_input, 1))
        else:
            for i in range(num_layers - 1):
                self.mlp.append(nn.Linear(d_input if i == 0 else d_hidden, d_hidden))
                self.mlp.append(nn.BatchNorm1d(d_hidden))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(d_hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for model in self.mlp:
            output = model(output)

        return output

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
