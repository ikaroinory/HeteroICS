from torch import Tensor, nn


class OutputLayer(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, num_layers: int):
        super().__init__()

        self.num_layers = num_layers

        self.mlp = nn.ModuleList()
        if num_layers == 1:
            self.mlp.append(nn.Linear(d_input, 1))
        else:
            self.res_layer = nn.Linear(d_input, 1)
            for i in range(num_layers - 1):
                self.mlp.append(nn.Linear(d_input if i == 0 else d_hidden, d_hidden))
                self.mlp.append(nn.BatchNorm1d(d_hidden))
                self.mlp.append(nn.LeakyReLU())
            self.mlp.append(nn.Linear(d_hidden, 1))

        self.init_params()

    def init_params(self):
        if self.num_layers == 1:
            nn.init.xavier_uniform_(self.res_layer.weight)
            nn.init.zeros_(self.res_layer.bias)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for model in self.mlp:
            output = model(output)

        return output if self.num_layers == 1 else output + self.res_layer(x)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
