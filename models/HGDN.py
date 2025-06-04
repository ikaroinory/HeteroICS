from collections import defaultdict

import torch
from torch import Tensor, nn

from data_types import NodeConfig, NodeType


class HGDN(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        d_hidden: int,
        d_output_hidden: int,
        num_heads: int,
        num_output_layer: int,
        k: int,
        *,
        node_config: dict[NodeType, NodeConfig],
        dtype=None,
        device=None
    ):
        super().__init__()

        self.node_indices_flatten_dict: dict[NodeType, Tensor] | None = None
        self.node_indices_dict: dict[NodeType, Tensor] = {
            node_type: torch.tensor(config['index'], dtype=torch.int32, device=device)
            for node_type, config in node_config.items()
        }
        self.node_index_to_type_dict: dict[int, NodeType] = {
            index.item(): node_type
            for node_type, node_indices in self.node_indices_dict.items()
            for index in node_indices
        }
        self.node_types = list(node_config.keys())

        num_nodes = sum(len(indices) for indices in self.node_indices_dict)
        self.num_nodes_dict = {node_type: len(indices) for node_type, indices in self.node_indices_dict.items()}

        self.k = k
        self.dtype = dtype
        self.device = device

        self.node_numbers = torch.arange(num_nodes).to(self.device)

        self.embedding_layer = nn.Sequential(
            nn.Embedding(num_embeddings=num_nodes, embedding_dim=d_hidden),
            nn.BatchNorm1d(d_hidden)
        )
        self.x_proj_layer_dict = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(sequence_len, d_hidden),
                nn.BatchNorm1d(d_hidden)
            )
            for node_type in self.node_types
        })
        self.v_proj_layer_dict = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.BatchNorm1d(d_hidden)
            )
            for node_type in self.node_types
        })

    @staticmethod
    def __cos_similarity(x: Tensor, y: Tensor) -> Tensor:
        x_norm = x.norm(dim=-1).unsqueeze(-1) + 1e-8
        y_norm = y.norm(dim=-1).unsqueeze(-1) + 1e-8

        return (x @ y.T) / (x_norm @ y_norm.T)

    def __get_edges(self, x: Tensor, y: Tensor, k: int) -> Tensor:
        similarity = self.__cos_similarity(x, y)

        _, indices = torch.topk(similarity, k, dim=-1)

        source_nodes = torch.arange(x.shape[0], device=self.device).repeat_interleave(k)
        target_nodes = indices.reshape(-1)

        edges = torch.stack([source_nodes, target_nodes], dim=0)

        return edges

    def __get_edge_index_dict(self, edges: Tensor) -> dict[tuple[str, str, str], Tensor]:
        edges_dict = defaultdict(list)
        for edge in edges.T:
            src_type = self.node_index_to_type_dict[edge[0].item()]
            dst_type = self.node_index_to_type_dict[edge[1].item()]
            edges_dict[(src_type, 'to', dst_type)].append(edge)

        edge_index_dict = {k: torch.stack(v, dim=-1) for k, v in edges_dict.items()}

        return edge_index_dict

    def __flatten(
        self,
        x: Tensor,
        v: Tensor,
        edge_index_dict: dict[tuple[NodeType, str, NodeType], Tensor]
    ) -> tuple[Tensor, Tensor, dict[tuple[NodeType, str, NodeType], Tensor], dict[NodeType, Tensor]]:
        batch_size, num_nodes, sequence_len = x.shape

        x_flatten = x.reshape(-1, sequence_len)

        _, d_hidden = v.shape
        v_flatten = v.repeat(batch_size, 1, 1).reshape(-1, d_hidden)

        steps = torch.arange(batch_size, device=self.device).to(x.device) * num_nodes
        steps_dict = {node_type: torch.arange(batch_size, device=self.device) * self.num_nodes_dict[node_type] for node_type in self.node_types}

        edge_index_flatten_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            edges = edge_index.repeat(1, batch_size)
            edges[0] += steps_dict[src_type].repeat_interleave(edges.shape[1] // batch_size)
            edges[1] += steps_dict[dst_type].repeat_interleave(edges.shape[1] // batch_size)

        node_indices_flatten_dict = {} if self.node_indices_flatten_dict is None else self.node_indices_flatten_dict
        if self.node_indices_flatten_dict is None:
            for node_type, node_indices in self.node_indices_dict.items():
                node_indices_flatten_dict[node_type] = node_indices.repeat(batch_size) + steps.repeat_interleave(node_indices.shape[0])

        return x_flatten, v_flatten, edge_index_flatten_dict, node_indices_flatten_dict

    def forward(self, x: Tensor):
        batch_size, num_nodes, _ = x.shape

        v = self.embedding_layer(self.node_numbers)
        edges = self.__get_edges(v, v, self.k)

        edge_index_dict = self.__get_edge_index_dict(edges)

        x_flatten, v_flatten, edge_index_flatten_dict, node_indices_flatten_dict = self.__flatten(x, v, edge_index_dict)
        x_proj_dict = {node_type: self.x_proj_layer_dict[node_type](x_flatten[node_indices_flatten_dict[node_type]]) for node_type in self.node_types}
        v_proj_dict = {node_type: self.v_proj_layer_dict[node_type](v_flatten[node_indices_flatten_dict[node_type]]) for node_type in self.node_types}
