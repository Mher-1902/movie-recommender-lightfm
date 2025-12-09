import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


# ---------- CONFIG ----------

# Path to MovieLens 20M CSVs
DATA_DIR = Path.home() / "data" / "ml-20m"

DB_USER = "mluser"
DB_PASSWORD = "mlpass"
DB_HOST = "localhost"
DB_PORT = 5433          # host port from docker compose
DB_NAME = "movielens"

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_movies(engine):
    print("Loading movies...")
    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found at {movies_path}")

    df = pd.read_csv(movies_path)

    # MovieLens columns: movieId, title, genres
    df = df.rename(columns={
        "movieId": "movie_id",
        "title": "title",
        "genres": "genres",
    })

    df.to_sql("movies", engine, if_exists="append", index=False)
    print(f"Inserted {len(df)} movies.")


def load_ratings(engine, chunksize=500_000):
    print("Loading ratings in chunks...")
    ratings_path = DATA_DIR / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found at {ratings_path}")

    total = 0
    for chunk in pd.read_csv(ratings_path, chunksize=chunksize):
        # MovieLens columns: userId, movieId, rating, timestamp
        chunk = chunk.rename(columns={
            "userId": "user_id",
            "movieId": "movie_id",
            "rating": "rating",
            "timestamp": "ts",
        })

        # parse datetime string → int64 (ns since epoch)
        chunk["ts"] = pd.to_datetime(chunk["ts"]).astype("int64")

        chunk.to_sql("ratings", engine, if_exists="append", index=False)
        total += len(chunk)
        print(f"  inserted {total} ratings so far...")

    print(f"Finished inserting {total} ratings.")


def load_tags(engine, chunksize=200_000):
    print("Loading tags in chunks...")
    tags_path = DATA_DIR / "tags.csv"
    if not tags_path.exists():
        print(f"tags.csv not found at {tags_path}, skipping tags.")
        return

    total = 0
    for chunk in pd.read_csv(tags_path, chunksize=chunksize):
        # MovieLens columns: userId, movieId, tag, timestamp
        chunk = chunk.rename(columns={
            "userId": "user_id",
            "movieId": "movie_id",
            "tag": "tag",
            "timestamp": "ts",
        })

        # keep only rows that actually have a tag string
        chunk = chunk.dropna(subset=["tag"])

        # parse datetime string → int64 (ns since epoch)
        chunk["ts"] = pd.to_datetime(chunk["ts"]).astype("int64")

        chunk.to_sql("tags", engine, if_exists="append", index=False)
        total += len(chunk)
        print(f"  inserted {total} tags so far...")

    print(f"Finished inserting {total} tags.")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print(f"Connecting to DB: {CONN_STR}")
    engine = create_engine(CONN_STR)

    # movies and ratings are already in the DB
    # load_movies(engine)
    # load_ratings(engine)
    load_tags(engine)

    print("Tags loaded successfully.")


if __name__ == "__main__":
    main()
