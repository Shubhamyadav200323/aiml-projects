"""GraphSAGE node classifier for fraud-ring membership.

Uses `torch_geometric.nn.SAGEConv` when PyTorch Geometric is installed. A
numerically equivalent mean-aggregation SAGE layer is provided so the model
(and CI) still runs on platforms where PyG wheels are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fraud.models.graph import AccountGraph
from fraud.models.metrics import evaluate

try:  # pragma: no cover - depends on environment
    from torch_geometric.nn import SAGEConv as _PygSAGEConv

    HAS_PYG = True
except Exception:  # pragma: no cover
    HAS_PYG = False


class MeanSAGEConv(nn.Module):
    """h_i' = W_self h_i + W_neigh mean_{j in N(i)} h_j   (Hamilton et al., 2017)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(agg)


def _conv(in_dim: int, out_dim: int, use_pyg: bool) -> nn.Module:
    return _PygSAGEConv(in_dim, out_dim) if (use_pyg and HAS_PYG) else MeanSAGEConv(in_dim, out_dim)


class FraudGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2,
                 use_pyg: bool = True) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(_conv(a, b, use_pyg) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + in_dim, 1)  # skip connection keeps raw node signal

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, edge_index))))
        return self.head(torch.cat([h, x], dim=-1)).squeeze(-1)


@dataclass
class GNNResult:
    model: FraudGraphSAGE
    scores: np.ndarray        # fraud probability per node
    metrics: dict[str, float]


def train_gnn(graph: AccountGraph, epochs: int = 200, lr: float = 0.01, hidden: int = 64,
              val_frac: float = 0.3, seed: int = 0, use_pyg: bool = True) -> GNNResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = graph.num_nodes
    # stratified node split so the rare positive class appears in validation
    pos, neg = np.where(graph.y == 1)[0], np.where(graph.y == 0)[0]
    val = np.concatenate([rng.choice(pos, int(len(pos) * val_frac), replace=False),
                          rng.choice(neg, int(len(neg) * val_frac), replace=False)])
    train_mask = np.ones(n, dtype=bool)
    train_mask[val] = False

    x = torch.from_numpy(graph.x)
    ei = torch.from_numpy(graph.edge_index)
    y = torch.from_numpy(graph.y)
    tm = torch.from_numpy(train_mask)
    model = FraudGraphSAGE(x.size(1), hidden=hidden, use_pyg=use_pyg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = (len(neg) / max(len(pos), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))

    best_state, best_ap = None, -1.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x, ei)[tm], y[tm])
        loss.backward()
        opt.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x, ei)).numpy()
            ap = evaluate(graph.y[~train_mask], p[~train_mask])["pr_auc"]
            if ap > best_ap:
                best_ap, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, ei)).numpy()
    metrics = evaluate(graph.y[~train_mask], scores[~train_mask])
    metrics["backend_pyg"] = float(use_pyg and HAS_PYG)
    return GNNResult(model, scores, metrics)


@torch.no_grad()
def score_graph(model: FraudGraphSAGE, graph: AccountGraph) -> np.ndarray:
    """Inductive inference: apply a trained model to a (newer / larger) graph."""
    model.eval()
    x = torch.from_numpy(graph.x)
    return torch.sigmoid(model(x, torch.from_numpy(graph.edge_index))).numpy()


def save_scores(graph: AccountGraph, scores: np.ndarray, path: str | Path) -> None:
    import pandas as pd

    pd.DataFrame({"user_id": graph.users, "graph_score": scores}).to_parquet(path, index=False)
