import torch, pandas as pd

class TemporalBatchLoader:
    """
    Streams interactions in chronological order.
    Returns tensors (u, i, y, t) for each mini-batch.
    """
    def __init__(self, csv_path, batch_size=4096):
        self.csv_path = csv_path
        self.batch_size = batch_size

    def __iter__(self):
        df = pd.read_csv(self.csv_path)  # already time-sorted in prep
        self.u = torch.tensor(df["user_id"].values,  dtype=torch.long)
        self.i = torch.tensor(df["movie_id"].values, dtype=torch.long)
        self.y = torch.tensor(df["label"].values,    dtype=torch.float32)
        self.t = torch.tensor(df["ts_norm"].values,  dtype=torch.float32).unsqueeze(1)
        self.n = len(self.u); self.pos = 0
        return self

    def __next__(self):
        if self.pos >= self.n: raise StopIteration
        j = min(self.pos+self.batch_size, self.n)
        batch = {
            "u": self.u[self.pos:j],
            "i": self.i[self.pos:j],
            "y": self.y[self.pos:j],
            "t": self.t[self.pos:j],
        }
        self.pos = j
        return batch
