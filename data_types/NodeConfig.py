from typing import Literal, NotRequired, TypedDict


class NodeConfig(TypedDict):
    value_type: Literal['float', 'enum']
    value_list: NotRequired[list[int]]
    index: list[int]
