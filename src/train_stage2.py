# -*- coding: utf-8 -*-
"""
Stage 2: Baseline heterogeneous GNN (GraphSAGE) for link prediction on MovieLens-1M.
- Loads hetero graphs from Stage 1 (graph_train.pt / graph_val.pt)
- Adds reverse edges (ToUndirected)
- Trains with negative sampling via LinkNeighborLoader
- Reports Precision@K / Recall@K on validation set
"""

import argparse
import math
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T


# -----------------------
# Utilities
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic-ish behavior on CPU
    torch.use_deterministic_algorithms(False)

def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_graph(path: str, device: str) -> HeteroData:
    # PyTorch >= 2.6: torch.load defaults to weights_only=True; we need the full object
    g: HeteroData = torch.load(path, weights_only=False)
    # Add reverse edges so messages can go both ways (user<->item, item<->genre)
    g = T.ToUndirected()(g)
    g = g.to(device)
    return g


# -----------------------
# Model
# -----------------------

class FeatEncoder(nn.Module):
    """
    Simple feature encoder:
      - expects every node type to have .x
      - projects to 'hidden' dims with a Linear
    """
    def __init__(self, g: HeteroData, hidden: int):
        super().__init__()
        self.proj = nn.ModuleDict()
        for ntype in g.node_types:
            in_dim = g[ntype].x.size(-1)
            self.proj[ntype] = nn.Linear(in_dim, hidden)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {ntype: F.relu(self.proj[ntype](x)) for ntype, x in x_dict.items()}


class HeteroSAGE(nn.Module):
    """
    Two-layer GraphSAGE over all edge types with shared HeteroConv blocks.
    Decoding = dot-product on (user,item) embeddings.
    """
    def __init__(self, g: HeteroData, hidden: int = 64):
        super().__init__()
        self.encoder = FeatEncoder(g, hidden)
        self.layers = nn.ModuleList()
        for _ in range(2):
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), hidden) for et in g.edge_types},
                aggr="sum"
            )
            self.layers.append(conv)
        self.target_et: Tuple[str, str, str] = ("user", "interacted", "item")

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        h = self.encoder(x_dict)
        for conv in self.layers:
            h = conv(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h.items()}
        return h

    def decode(self, h_user: torch.Tensor, h_item: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        edge_label_index: shape [2, E] with (user_idx, item_idx)
        Returns logits (unnormalized scores) shape [E]
        """
        src, dst = edge_label_index
        return (h_user[src] * h_item[dst]).sum(dim=-1)


# -----------------------
# Data loaders
# -----------------------

def make_loader(g: HeteroData, batch_size: int = 4096, neg_ratio: float = 1.0, shuffle: bool = True) -> LinkNeighborLoader:
    et = ("user", "interacted", "item")
    return LinkNeighborLoader(
        g,
        num_neighbors={k: [15, 10] for k in g.edge_types},  # 2-hop sampling
        batch_size=batch_size,
        edge_label_index=(et, g[et].edge_index),
        neg_sampling_ratio=neg_ratio,   # 1 negative per positive
        shuffle=shuffle,
    )


# -----------------------
# Metrics
# -----------------------

@torch.no_grad()
def precision_recall_at_k(model: nn.Module, g_val: HeteroData, k: int = 10, device: str = "cpu") -> Tuple[float, float]:
    """
    Computes a simple batch-level Precision@K and Recall@K across LinkNeighborLoader minibatches.
    Note: This is a *quick baseline*. For rigorous metrics, you would do per-user ranking over all items.
    """
    model.eval()
    loader = make_loader(g_val, batch_size=8192, neg_ratio=1.0, shuffle=False)

    tp, fp, fn = 0, 0, 0
    et = ("user", "interacted", "item")

    for batch in loader:
        batch = batch.to(device)
        h = model(batch.x_dict, batch.edge_index_dict)
        scores = model.decode(h["user"], h["item"], batch[et].edge_label_index)
        labels = batch[et].edge_label  # 1 for real edges, 0 for negatives

        # Rank within the micro-batch and pick top-k
        kk = min(k, scores.numel())
        topk_idx = torch.topk(scores, k=kk).indices

        preds = torch.zeros_like(labels, dtype=torch.bool)
        preds[topk_idx] = True

        tp += (preds & (labels == 1)).sum().item()
        fp += (preds & (labels == 0)).sum().item()
        fn += ((~preds) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(precision), float(recall)


# -----------------------
# Training
# -----------------------

def train(args):
    set_seed(args.seed)
    device = device_str()
    print(f"[Info] Device: {device}")

    # Load graphs
    g_train = load_graph(args.train_graph, device)
    g_val   = load_graph(args.val_graph, device)

    # Build model
    model = HeteroSAGE(g_train, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.BCEWithLogitsLoss()

    # Loader
    train_loader = make_loader(g_train, batch_size=args.batch_size, neg_ratio=args.neg_ratio, shuffle=True)
    et = ("user", "interacted", "item")

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            h = model(batch.x_dict, batch.edge_index_dict)
            logits = model.decode(h["user"], h["item"], batch[et].edge_label_index)
            loss = loss_fn(logits, batch[et].edge_label.float())
            loss.backward()
            opt.step()
            running += loss.item()

        # quick val metric
        prec, rec = precision_recall_at_k(model, g_val, k=args.topk, device=device)
        print(f"Epoch {epoch:02d} | loss={running/len(train_loader):.4f} | P@{args.topk}={prec:.4f} R@{args.topk}={rec:.4f}")

    # Save weights
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    torch.save(model.state_dict(), args.out)
    print(f"[OK] Saved model → {args.out}")


# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Baseline GraphSAGE training for link prediction")
    p.add_argument("--train_graph", type=str, default="data/processed/graph_train.pt")
    p.add_argument("--val_graph",   type=str, default="data/processed/graph_val.pt")
    p.add_argument("--out",         type=str, default="data/processed/graphsage_baseline.pt")
    p.add_argument("--hidden",      type=int, default=64)
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--batch_size",  type=int, default=4096)
    p.add_argument("--neg_ratio",   type=float, default=1.0)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--wd",          type=float, default=1e-5)
    p.add_argument("--topk",        type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
