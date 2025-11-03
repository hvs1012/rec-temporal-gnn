import math, torch, json
import torch.nn as nn
from pathlib import Path
from temporal_loader import TemporalBatchLoader
from model_tgat import TemporalTGAT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED = Path("data/processed")

# counts for embedding sizes
counts = json.loads((PROCESSED/"temporal_counts.json").read_text())
NUM_USERS = counts["num_users"]
NUM_ITEMS = counts["num_items"]

def precision_recall_at_k(scores, labels, k=10):
    # scores/labels grouped per-user within the batch
    # inputs are flattened over user-item pairs; we re-group by user index
    # Here we assume scores/labels are stacked per (u, pos+negs) in same order.
    # For simplicity, compute global top-k approximation:
    k = min(k, scores.numel())
    topk = torch.topk(scores, k=k).indices
    prec = labels[topk].float().mean().item()
    # simple proxy recall: TP / total positives (avoid per-user grouping for speed)
    pos = labels.sum().item()
    rec = float(labels[topk].sum().item() / pos) if pos > 0 else 0.0
    return prec, rec

def train_epoch(model, loader, neg_k=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        u = batch["u"].to(DEVICE)    # [E]
        i = batch["i"].to(DEVICE)    # [E]
        y = batch["y"].to(DEVICE)    # [E] in {0,1}
        t = batch["t"].to(DEVICE)    # [E,1] float32

        # keep only positives to generate proper negatives per user
        pos_mask = (y > 0.5)
        if pos_mask.sum() == 0:
            continue
        u_pos, i_pos, t_pos = u[pos_mask], i[pos_mask], t[pos_mask]   # [P], [P], [P,1]

        # --- NEGATIVE SAMPLING (same users, random items) ---
        # For each positive, sample neg_k items uniformly
        num_pos = u_pos.numel()
        neg_items = torch.randint(0, NUM_ITEMS, (num_pos*neg_k,), device=DEVICE)
        u_neg = u_pos.repeat_interleave(neg_k)
        t_neg = t_pos.repeat_interleave(neg_k, dim=0)

        # --- BUILD LOCAL MINI-GRAPH (relabel ids) ---
        u_all = torch.cat([u_pos, u_neg], dim=0)
        i_all = torch.cat([i_pos, neg_items], dim=0)
        t_all = torch.cat([t_pos, t_neg], dim=0)
        lbl   = torch.cat([torch.ones_like(u_pos, dtype=torch.float32),
                           torch.zeros_like(u_neg, dtype=torch.float32)], dim=0)

        u_unique, u_inv = torch.unique(u_all, return_inverse=True)
        i_unique, i_inv = torch.unique(i_all, return_inverse=True)
        U, I = u_unique.size(0), i_unique.size(0)

        src_local = u_inv                     # [E']
        dst_local = i_inv + U                 # [E']
        edge_index = torch.stack([src_local, dst_local], dim=0)

        # --- FORWARD ---
        out = model(u_unique, i_unique, edge_index, t_all)
        h_user = out[:U]
        h_item = out[U:]
        logits = (h_user[src_local] * h_item[dst_local - U]).sum(dim=-1)
        logits = logits / math.sqrt(h_user.size(-1))     # scale for stability

        # --- LOSS ---
        loss = bce(logits, lbl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def eval_epoch(model, loader, neg_k=50):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            u = batch["u"].to(DEVICE)
            i = batch["i"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            t = batch["t"].to(DEVICE)

            pos_mask = (y > 0.5)
            if pos_mask.sum() == 0:
                continue
            u_pos, i_pos, t_pos = u[pos_mask], i[pos_mask], t[pos_mask]

            num_pos = u_pos.numel()
            neg_items = torch.randint(0, NUM_ITEMS, (num_pos*neg_k,), device=DEVICE)
            u_neg = u_pos.repeat_interleave(neg_k)
            t_neg = t_pos.repeat_interleave(neg_k, dim=0)

            u_all = torch.cat([u_pos, u_neg], dim=0)
            i_all = torch.cat([i_pos, neg_items], dim=0)
            t_all = torch.cat([t_pos, t_neg], dim=0)
            lbl   = torch.cat([torch.ones_like(u_pos, dtype=torch.float32),
                               torch.zeros_like(u_neg, dtype=torch.float32)], dim=0)

            u_unique, u_inv = torch.unique(u_all, return_inverse=True)
            i_unique, i_inv = torch.unique(i_all, return_inverse=True)
            U = u_unique.size(0)

            src_local = u_inv
            dst_local = i_inv + U
            edge_index = torch.stack([src_local, dst_local], dim=0)

            out = model(u_unique, i_unique, edge_index, t_all)
            h_user = out[:U]; h_item = out[U:]
            logits = (h_user[src_local] * h_item[dst_local - U]).sum(dim=-1)
            logits = logits / math.sqrt(h_user.size(-1))
            probs  = torch.sigmoid(logits)

            all_scores.append(probs.cpu())
            all_labels.append(lbl.cpu())

    if not all_scores:
        return 0.0, 0.0
    scores = torch.cat(all_scores); labels = torch.cat(all_labels)
    return precision_recall_at_k(scores, labels, k=10)

if __name__ == "__main__":
    train_loader = TemporalBatchLoader(str(PROCESSED/"temporal_train.csv"), batch_size=4096)
    val_loader   = TemporalBatchLoader(str(PROCESSED/"temporal_val.csv"),   batch_size=4096)

    model = TemporalTGAT(NUM_USERS, NUM_ITEMS, hidden=64, time_dim=32).to(DEVICE)

    for epoch in range(1, 6):
        tr_loss = train_epoch(model, train_loader, neg_k=1, lr=5e-4)
        p10, r10 = eval_epoch(model, val_loader, neg_k=50)
        print(f"Epoch {epoch:02d} | loss={tr_loss:.4f} | P@10={p10:.4f} R@10={r10:.4f}")

    torch.save(model.state_dict(), PROCESSED/"tgat_baseline.pt")
    print("[OK] Saved:", PROCESSED/"tgat_baseline.pt")
