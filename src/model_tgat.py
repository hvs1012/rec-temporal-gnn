# src/model_tgat.py
# --------------------------------------------------------------
# Stage 3: Temporal Graph Attention Network (TGAT-style encoder)
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor):  # t: [E,1]
        t = t.to(torch.float32)
        d = self.dim
        freq = torch.arange(d, device=t.device, dtype=t.dtype) / float(d)
        scales = 1.0 / (10.0 ** freq)
        x = t * scales.unsqueeze(0)
        out = torch.sin(x)
        return self.lin(out)  # [E, dim]


class TemporalTGAT(nn.Module):
    """
    TGAT-style bipartite encoder:
      • user/item embeddings
      • time encodings as edge_attr in TransformerConv
    """

    def __init__(self, num_users, num_items, hidden=64, time_dim=32):
        super().__init__()
        # ID embeddings
        self.user_emb = nn.Embedding(num_users, hidden)
        self.item_emb = nn.Embedding(num_items, hidden)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

        # time encoder + temporal attention layer
        self.time_enc = TimeEncoder(time_dim)
        self.conv = TransformerConv(
            in_channels=hidden,
            out_channels=hidden,
            heads=2,
            dropout=0.1,
            edge_dim=time_dim,  # critical: time as edge_attr
        )

    def forward(self, u_unique, i_unique, edge_index, t_edge):
        """
        u_unique: [U] user ids (global)
        i_unique: [I] item ids (global)
        edge_index: [2,E] local ids
        t_edge: [E,1] float timestamps
        """
        # node features from embeddings
        x_user = self.user_emb(u_unique)
        x_item = self.item_emb(i_unique)
        h = torch.cat([x_user, x_item], dim=0)

        # time encodings -> edge_attr
        eattr = self.time_enc(t_edge)

        out = self.conv(h, edge_index, edge_attr=eattr)
        return out
