from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Tuple

# 这部分代码定义的是中间图表示，还不是GNN，也就是把CPN的语义层转为了图结构
# 核心是三个对象：GraphNode决定图节点信息，GraphEdge决定图中的边属性，TokenBindingGraph则相当于一个完整的神经观察快照

@dataclass
class GraphNode:
    """One typed node with learnable features separated from debug identity."""

    node_id: str    # 图内部用于找节点的稳定键，决定这枚图节点是谁，注意目前还是图表示，还没到接入网络的地步
    node_type: str  # 异构节点的类别，用于区分节点是工序、机器、Job、合法边、工业顺序、事件、库所、变迁、Binding
    numeric: Dict[str, float] = field(default_factory=dict)     # 数值特征，这是正式接入连续神经网络的数值特征，比如Processing_time、Failure_count等
    categorical: Dict[str, str] = field(default_factory=dict)   # 这是离散语义，比如Current_place、Event_type等，跟前面的连续数值语义不一样
    identity: Dict[str, Any] = field(default_factory=dict)      # 重要字段，用于寻找引用对象、Debug、恢复语义、建立关系边，其关系进入神经网络，而不是数值

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "numeric": dict(self.numeric),
            "categorical": dict(self.categorical),
            "identity": _json_value(self.identity),
        }


@dataclass
class GraphEdge:
    """A directed semantic relation between two graph nodes."""

    source: str     # 起始点
    target: str     # 终点
    edge_type: str  # 边类型
    numeric: Dict[str, float] = field(default_factory=dict)     # 边权重
    categorical: Dict[str, str] = field(default_factory=dict)   # 边的离散语义

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "numeric": dict(self.numeric),
            "categorical": dict(self.categorical),
        }

# 这是一个完整的神经观察快照，相当于保存time时刻的网络状态
@dataclass
class TokenBindingGraph:
    """Framework-neutral CPN observation with variable node and action counts."""

    time: float
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    # 此处的Action_keys需要注意，他会把图中的Binding节点翻译为CPN的Binding Key，比如
    # binding:000003
    #     ↓
    # (T_StartOperation, O12, M3, E_O12_M3)
    # 因此GNN最终只需要选择Binding: 000003，就可以反查原始CPN动作并激发，他实现了从图结构到实际PCN动作的翻译，而CPN动作跟FJSP环境是完整对应的
    action_keys: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate graph node: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def nodes_of_type(self, node_type: str) -> List[GraphNode]:
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def edges_of_type(self, edge_type: str) -> List[GraphEdge]:
        return [edge for edge in self.edges if edge.edge_type == edge_type]

    @property
    def action_node_ids(self) -> Tuple[str, ...]:
        return tuple(self.action_keys)

    # 这个函数可以理解为图观察的不变量检查
    # 他会检查：
    # graph time必须finite；
    # 所有numeric feature必须finite；
    # 每条边的source和target必须真实存在；
    # action map不能指向不存在的节点；
    # action节点必须真的是binding；
    # Binding key不能重复。
    def validate(self) -> None:
        if not math.isfinite(float(self.time)):
            raise ValueError("Graph time must be finite")
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"Node dictionary key mismatch: {node_id}")
            for feature_name, value in node.numeric.items():
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"Non-finite node feature {node_id}.{feature_name}={value}"
                    )
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(
                    f"Dangling graph edge {edge.edge_type}: "
                    f"{edge.source} -> {edge.target}"
                )
            for feature_name, value in edge.numeric.items():
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"Non-finite edge feature {edge.edge_type}.{feature_name}={value}"
                    )
        if set(self.action_keys) - set(self.nodes):
            raise ValueError("Action map refers to missing Binding nodes")
        for node_id in self.action_keys:
            if self.nodes[node_id].node_type != "binding":
                raise ValueError(f"Action node is not a Binding: {node_id}")
        if len({repr(key) for key in self.action_keys.values()}) != len(self.action_keys):
            raise ValueError("Duplicate Binding action keys")

    def summary(self) -> Dict[str, Any]:
        node_counts: Dict[str, int] = {}
        edge_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
        for edge in self.edges:
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1
        return {
            "time": float(self.time),
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "actions": len(self.action_keys),
            "node_counts": dict(sorted(node_counts.items())),
            "edge_counts": dict(sorted(edge_counts.items())),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": float(self.time),
            "summary": self.summary(),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "actions": [
                {
                    "node_id": node_id,
                    "binding_key": _json_value(binding_key),
                }
                for node_id, binding_key in self.action_keys.items()
            ],
            "metadata": _json_value(self.metadata),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
