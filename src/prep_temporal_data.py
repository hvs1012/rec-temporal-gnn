import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor):  # t: [E,1] float32
        t = t.to(torch.float32)
        d = self.dim
        freq = torch.arange(d, device=t.device, dtype=t.dtype) / float(d)
        scales = 1.0 / (10.0 ** freq)       # [d]
        x = t * scales.unsqueeze(0)         # [E,d]
        return self.lin(torch.sin(x))       # [E,d]

class TemporalTGAT(nn.Module):
    """
    Bipartite TGAT:
      - global nn.Embedding tables for users & items
      - per-batch local relabeling
      - temporal info as edge_attr to TransformerConv
    """
    def __init__(self, num_users, num_items, hidden=64, time_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden)
        self.item_emb = nn.Embedding(num_items, hidden)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

        self.time_enc = TimeEncoder(time_dim)
        self.conv = TransformerConv(
            in_channels=hidden,
            out_channels=hidden,
            heads=2,
            dropout=0.1,
            edge_dim=time_dim,               # time goes here
        )

    def forward(self, u_unique, i_unique, edge_index, t_edge):
        """
        u_unique: [U] global user ids
        i_unique: [I] global item ids
        edge_index: [2,E] with local ids (0..U-1, U..U+I-1)
        t_edge: [E,1] timestamps float32
        """
        x_user = self.user_emb(u_unique)     # [U,hidden]
        x_item = self.item_emb(i_unique)     # [I,hidden]
        h = torch.cat([x_user, x_item], dim=0)  # [U+I, hidden]
        eattr = self.time_enc(t_edge)           # [E, time_dim]
        out = self.conv(h, edge_index, edge_attr=eattr)  # [U+I, hidden]
        return out
