from collections import defaultdict

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax


class HAN(MessagePassing):
    def __init__(
        self,
        x_input: int,
        d_output: int,
        num_heads: int,
        *,
        node_indices: dict[str, list[int]],
        edge_types: list[tuple[str, str, str]]
    ):
        super().__init__(aggr='add', node_dim=0)

        self.node_types = list(node_indices.keys())
        self.d_output = d_output
        self.num_heads = num_heads

        self.W_x_phi = nn.ModuleDict({node_type: nn.Linear(x_input, d_output, bias=False) for node_type in self.node_types})
        self.w_pi = nn.ParameterDict(
            {
                '->'.join(edge_type): nn.Parameter(torch.zeros([1, num_heads, (d_output // num_heads) * 4]))
                for edge_type in edge_types
            }
        )
        # self.semantic_attention = nn.Sequential(
        #     nn.Linear(d_output, d_output),
        #     nn.Tanh(),
        #     nn.Linear(d_output, 1, bias=False)
        # )
        self.softmax = nn.Softmax(dim=0)
        self.W_beta = nn.Parameter(torch.zeros([d_output, d_output]))
        glorot(self.W_beta)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x_dict: dict[str, Tensor], v_dict: dict[str, Tensor], edge_index_dict: dict[tuple[str, str, str], Tensor]) -> dict[str, Tensor]:
        x_prime_dict = {}
        for node_type, x in x_dict.items():
            # x: [num_nodes * batch_size, sequence_len]
            x_prime_dict[node_type] = self.W_x_phi[node_type](x)  # [num_nodes, d_output]

        z_list_dict: dict[str, list[Tensor]] = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            # print(g_dict[src_type][edge_index[0]])  # xj
            # print(g_dict[dst_type][edge_index[1]])  # xi
            z: Tensor = self.propagate(
                edge_index,
                x=(x_prime_dict[src_type], x_prime_dict[dst_type]),
                v=(v_dict[src_type], v_dict[dst_type]),
                edge_type=edge_type
            )
            z = self.leaky_relu(z)  # [num_nodes, d_output]

            z_list_dict[dst_type].append(z)

        z_dict = {}
        for node_type, z_list in z_list_dict.items():
            # z_all = torch.stack(tuple(z_list), dim=0)
            #
            # beta = self.semantic_attention(z_all).mean(dim=1).squeeze()
            # beta = self.softmax(beta)
            #
            # output = torch.sum(beta.view(-1, 1, 1) * z_all, dim=0)
            #
            # z_dict[node_type] = output

            z_all = torch.stack(tuple(z_list), dim=0)
            beta = (z_all @ self.W_beta @ v_dict[node_type].T).diagonal(dim1=1, dim2=2).unsqueeze(-1)
            beta = self.softmax(beta)
            output = torch.sum(beta.expand(-1, -1, self.d_output) * z_all, dim=0)
            z_dict[node_type] = output / len(z_list)

        return z_dict

    def message(self, x_j: Tensor, x_i: Tensor, v_j: Tensor, v_i: Tensor, edge_index_i: Tensor, edge_type: tuple[str, str, str]) -> Tensor:
        x_i_heads = x_i.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        x_j_heads = x_j.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        v_i_heads = v_i.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        v_j_heads = v_j.reshape(-1, self.num_heads, self.d_output // self.num_heads)

        g_i = torch.cat([v_i_heads, x_i_heads], dim=-1)  # [num_nodes, num_heads, (d_output // num_heads) * 2]
        g_j = torch.cat([v_j_heads, x_j_heads], dim=-1)
        g = torch.cat([g_i, g_j], dim=-1)  # [num_nodes, num_heads, (d_output // num_heads) * 4]

        edge_type_str = '->'.join(edge_type)

        pi = self.leaky_relu(torch.einsum('nhd,nhd->nh', g, self.w_pi[edge_type_str]))
        alpha = softmax(pi, index=edge_index_i)

        return (alpha.view(-1, self.num_heads, 1) * x_j_heads).reshape(-1, self.d_output)

    def __call__(self, x_dict: dict[str, Tensor], v_dict: dict[str, Tensor], edge_index_dict: dict[tuple[str, str, str], Tensor]) -> dict[str, Tensor]:
        return super().__call__(x_dict, v_dict, edge_index_dict)
