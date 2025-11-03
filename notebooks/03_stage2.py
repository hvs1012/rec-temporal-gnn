import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", DEVICE)

# 1) Load graphs (Stage 1 outputs)
g_train: HeteroData = torch.load("data/processed/graph_train.pt", weights_only=False)
print("Node types:", g_train.node_types)
print("Edge types:", g_train.edge_types)
for nt in g_train.node_types:
    print(f"{nt}.x:", tuple(g_train[nt].x.shape))
for et in g_train.edge_types:
    print(f"{et} edge_index:", tuple(g_train[et].edge_index.shape))

# 2) Define the same model skeleton used in train_stage2.py
class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, g: HeteroData, hidden=64):
        super().__init__()
        self.enc = torch.nn.ModuleDict({
            "user":  torch.nn.Linear(g["user"].x.size(-1),  hidden),
            "item":  torch.nn.Linear(g["item"].x.size(-1),  hidden),
            "genre": torch.nn.Linear(g["genre"].x.size(-1), hidden),
        })
        self.conv1 = HeteroConv({et: SAGEConv((-1,-1), hidden) for et in g.edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((-1,-1), hidden) for et in g.edge_types}, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: F.relu(self.enc[k](x)) for k, x in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

model = HeteroGraphSAGE(g_train, hidden=64).to(DEVICE)

# 3) Load the trained weights
ckpt = "data/processed/graphsage_baseline.pt"
state = torch.load(ckpt, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("✅ Loaded model:", ckpt)

# 4) Tiny forward on a 10-node slice per type (very fast)
subset_x = {nt: g_train[nt].x[:10].to(DEVICE) for nt in g_train.node_types}
subset_edges = {et: g_train[et].edge_index[:, :50].to(DEVICE) for et in g_train.edge_types}
with torch.no_grad():
    h = model(subset_x, subset_edges)
print("Embeddings shapes:", {k: tuple(v.shape) for k, v in h.items()})

print("✅ Quick sanity passed.")
