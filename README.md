

| Stage | Description | Status |
|:------|:-------------|:--------|
| **Stage 1** | Data ingestion, preprocessing & graph construction | Completed |
| **Stage 2** | Baseline heterogeneous GNN (GraphSAGE), link prediction & evaluation | Completed |
| **Stage 3** | Temporal / attention-based GNN enhancement | ⏳ Working on it|

---

---

### 🧩 Stage 3 — Temporal Graph Attention Network (TGAT)

**Objective:**  
Extend the static GraphSAGE recommender into a **temporal graph model** that incorporates interaction timestamps through temporal attention.

**Key Components**
| Component | Description |
|------------|--------------|
| `prep_temporal_data.py` | Converts MovieLens ratings into chronological train/val/test splits |
| `temporal_loader.py` | Streams interactions in time order (continuous updates) |
| `model_tgat.py` | TGAT model with `TransformerConv(edge_dim=time_encoding)` |
| `train_stage3.py` | Trains temporal model, evaluates with Precision@K / Recall@K |
| `stage3.ipynb` | Validation notebook — confirms model convergence and per-user ranking accuracy |

**Results (baseline)**  
| Metric | Static GNN | TGAT (temporal) |
|---------|-------------|----------------|
| Precision@10 | 0.17 | **0.20** |
| Recall@10 | 0.01 | **0.05** |
| Loss | 0.69 → 0.45 | 0.69 → 0.45 |
> TGAT learns to emphasize recent interactions, showing improved temporal consistency and next-item ranking accuracy.

---


