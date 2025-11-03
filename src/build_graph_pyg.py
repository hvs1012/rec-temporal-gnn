from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import HeteroData

PROCESSED = Path("data/processed")
RAW = Path("data/raw/ml-1m")

def load_id_maps():
    user_map = pd.read_csv(PROCESSED/"user_map.csv")   # cols: user_id, raw_user_id
    item_map = pd.read_csv(PROCESSED/"item_map.csv")   # cols: movie_id, raw_movie_id
    return user_map, item_map

def build_genre_table(item_map: pd.DataFrame):
    # movies.dat: MovieID::Title::Genres (Genres separated by '|')
    movies = pd.read_csv(
    RAW / "movies.dat",
    sep="::",
    engine="python",
    names=["raw_movie_id", "title", "genres"],
    encoding="latin-1"          # <-- added this as some values contains non-UTF8 chars
    # errors="ignore",          
)

    # Join raw→encoded
    movies = movies.merge(item_map, on="raw_movie_id", how="inner")  # brings 'movie_id' (encoded)
    # Explode genres into rows: one (movie_id, genre) per relation
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies = movies.assign(genre_list=movies["genres"].str.split("|")).explode("genre_list")
    movies = movies.rename(columns={"genre_list":"genre"})
    return movies[["movie_id","genre"]]

def compute_node_features(ratings_all: pd.DataFrame):
    # user features: interaction count
    u_feat = ratings_all.groupby("user_id").size().rename("num_interactions").astype("int32")

    # item features: avg rating + interaction count
    it_cnt = ratings_all.groupby("movie_id").size().rename("num_interactions").astype("int32")
    it_avg = ratings_all.groupby("movie_id")["rating"].mean().rename("avg_rating").astype("float32")
    item_feat = pd.concat([it_avg, it_cnt], axis=1).fillna(0.0)

    return u_feat, item_feat

def build_hetero_for_split(split_csv: Path, movies_genres: pd.DataFrame,
                           u_feat: pd.Series, item_feat: pd.DataFrame) -> HeteroData:
    df = pd.read_csv(split_csv)  # cols: user_id, movie_id, rating, timestamp

    # counts
    num_users = int(df["user_id"].max()) + 1
    num_items = int(df["movie_id"].max()) + 1

    # ----- nodes -----
    data = HeteroData()
    data["user"].num_nodes = num_users
    data["item"].num_nodes = num_items

    # user.x (num_interactions) — compute from full ratings_all so it's stable
    u_x = torch.zeros((num_users, 1), dtype=torch.float32)
    u_sub = u_feat.reindex(range(num_users)).fillna(0).to_numpy().reshape(-1,1)
    u_x[:] = torch.tensor(u_sub, dtype=torch.float32)
    data["user"].x = u_x

    # item.x ([avg_rating, num_interactions])
    it = item_feat.reindex(range(num_items)).fillna(0.0)
    it_x = torch.tensor(it.to_numpy(), dtype=torch.float32)
    data["item"].x = it_x

    # ----- user->item edges -----
    src = torch.tensor(df["user_id"].values, dtype=torch.long)
    dst = torch.tensor(df["movie_id"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    data[("user","interacted","item")].edge_index = edge_index

    # edge features
    data[("user","interacted","item")].edge_attr = torch.tensor(df[["rating"]].values, dtype=torch.float32)
    data[("user","interacted","item")].timestamp = torch.tensor(df["timestamp"].values, dtype=torch.int64)

    # ----- item->genre edges -----
    # Build a genre index
    unique_genres = sorted(movies_genres["genre"].dropna().unique().tolist())
    genre2id = {g:i for i,g in enumerate(unique_genres)}
    data["genre"].num_nodes = len(unique_genres)
    # simple genre node feature: popularity (how many items tagged)
    g_counts = movies_genres.groupby("genre").size().reindex(unique_genres).fillna(0).astype("int32")
    data["genre"].x = torch.tensor(g_counts.to_numpy().reshape(-1,1), dtype=torch.float32)

    # edges (item -> genre)
    mg = movies_genres.dropna().copy()
    mg["genre_id"] = mg["genre"].map(genre2id)
    # keep only item ids within current split num_items (safe if you used same encoding)
    mg = mg[mg["movie_id"] < num_items]
    e_src = torch.tensor(mg["movie_id"].values, dtype=torch.long)
    e_dst = torch.tensor(mg["genre_id"].values, dtype=torch.long)
    data[("item","belongs_to","genre")].edge_index = torch.stack([e_src, e_dst], dim=0)

    return data

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # Load maps and full ratings to compute stable node features
    user_map, item_map = load_id_maps()
    ratings_all = pd.read_csv(PROCESSED/"ratings_all.csv")
    movies_genres = build_genre_table(item_map)
    u_feat, item_feat = compute_node_features(ratings_all)

    for split in ["train","val","test"]:
        g = build_hetero_for_split(PROCESSED/f"ratings_{split}.csv",
                                   movies_genres, u_feat, item_feat)
        out = PROCESSED/f"graph_{split}.pt"
        torch.save(g, out)
        num_edges = g[("user","interacted","item")].edge_index.size(1)
        print(f"[OK] {split}: users={g['user'].num_nodes} items={g['item'].num_nodes} "
              f"genres={g['genre'].num_nodes} edges_ui={num_edges} → {out}")

if __name__ == "__main__":
    main()
