import argparse
from pathlib import Path
import pandas as pd
import yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def encode_and_save_maps(df, user_col="user_id", item_col="movie_id", out_dir=Path("data/processed")):
    # Use pandas.factorize → fast, deterministic, and gives us the inverse maps
    u_codes, u_uniques = pd.factorize(df[user_col], sort=True)   # u_codes are 0..(n_users-1)
    i_codes, i_uniques = pd.factorize(df[item_col], sort=True)   # i_codes are 0..(n_items-1)

    df[user_col] = u_codes
    df[item_col] = i_codes

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw→encoded maps for later joins (metadata, inference)
    pd.DataFrame({"raw_user_id": u_uniques}).reset_index(names="user_id").to_csv(out_dir/"user_map.csv", index=False)
    pd.DataFrame({"raw_movie_id": i_uniques}).reset_index(names="movie_id").to_csv(out_dir/"item_map.csv", index=False)

    return df

def temporal_split(df, train=0.8, val=0.1, test=0.1):
    assert abs(train + val + test - 1.0) < 1e-6
    n = len(df)
    t_end = int(train * n)
    v_end = int((train + val) * n)
    return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]

def ingest_movielens_1m(cfg):
    raw_dir = Path(cfg["raw_dir"])
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(raw_dir/"ratings.dat", sep="::", engine="python",
                          names=cfg["columns"]["ratings"])
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    # Encode IDs and save raw→encoded maps
    ratings = encode_and_save_maps(ratings, "user_id", "movie_id", out_dir)

    # Save full cleaned
    ratings.to_csv(out_dir/"ratings_all.csv", index=False)

    # Temporal split by index after chronological sort
    tr, va, te = temporal_split(ratings,
                                train=cfg["splits"]["train"],
                                val=cfg["splits"]["val"],
                                test=cfg["splits"]["test"])
    tr.to_csv(out_dir/"ratings_train.csv", index=False)
    va.to_csv(out_dir/"ratings_val.csv", index=False)
    te.to_csv(out_dir/"ratings_test.csv", index=False)

    # quick sanity
    print("[OK] Data saved to", out_dir)
    print({"num_users": ratings["user_id"].nunique(),
           "num_items": ratings["movie_id"].nunique(),
           "num_interactions": len(ratings)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ingest_movielens_1m(cfg)

if __name__ == "__main__":
    main()
