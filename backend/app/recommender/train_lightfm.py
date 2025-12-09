"""
Train a LightFM model on the MovieLens data stored in Postgres.

This script:
- Connects to the DB
- Builds a LightFM Dataset from ratings (user_id, movie_id, rating)
- Trains a WARP model
- Saves model + user/item ID mappings to disk

Run from project root:
    python backend/app/recommender/train_lightfm.py
"""

from pathlib import Path
import pickle

from sqlalchemy import create_engine, text
from lightfm.data import Dataset
from lightfm import LightFM


# ---------- CONFIG ----------

DB_USER = "mluser"
DB_PASSWORD = "mlpass"
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "movielens"

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Where to save model + mappings
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "lightfm_model.pkl"
USER_MAPPING_PATH = ARTIFACTS_DIR / "user_id_map.pkl"
ITEM_MAPPING_PATH = ARTIFACTS_DIR / "item_id_map.pkl"


def get_engine():
    print(f"Connecting to DB: {CONN_STR}")
    return create_engine(CONN_STR)


def get_unique_ids(engine):
    """
    Fetch distinct user_ids and movie_ids from ratings table using raw SQL.
    """
    print("Fetching distinct user_ids...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT user_id FROM ratings"))
        user_ids = [str(row[0]) for row in result]
    print(f"  found {len(user_ids)} users")

    print("Fetching distinct movie_ids...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT movie_id FROM ratings"))
        item_ids = [str(row[0]) for row in result]
    print(f"  found {len(item_ids)} items")

    return user_ids, item_ids


def ratings_triplets(engine):
    """
    Generator that yields (user_id, item_id, rating) triplets
    for LightFM's Dataset.build_interactions(), streaming directly
    from the database without pandas.
    """
    print("Streaming ratings from DB...")
    query = text("SELECT user_id, movie_id, rating FROM ratings")

    with engine.connect() as conn:
        # stream_results=True avoids loading everything into memory
        result = conn.execution_options(stream_results=True).execute(query)
        for row in result:
            # row has attributes user_id, movie_id, rating
            yield (str(row.user_id), str(row.movie_id), float(row.rating))


def build_dataset_and_interactions(engine):
    """
    Build a LightFM Dataset and the interactions/weights matrices.
    """
    user_ids, item_ids = get_unique_ids(engine)

    print("Fitting LightFM Dataset with users & items...")
    dataset = Dataset()
    dataset.fit(
        users=user_ids,
        items=item_ids,
    )

    print("Building interactions matrix from ratings...")
    interactions, weights = dataset.build_interactions(
        ratings_triplets(engine)
    )

    print("Interactions shape:", interactions.shape)
    return dataset, interactions, weights


def train_lightfm(interactions, weights, no_components=64, epochs=20, num_threads=4):
    """
    Train a LightFM model on the given interactions.
    """
    print("Training LightFM model...")
    model = LightFM(
        no_components=no_components,
        loss="warp",      # good for implicit-style ranking
    )

    model.fit(
        interactions,
        sample_weight=weights,
        epochs=epochs,
        num_threads=num_threads,
    )

    print("Training complete.")
    return model


def save_artifacts(model, dataset):
    """
    Save the LightFM model and ID mappings to disk.
    """
    print(f"Saving model to {MODEL_PATH} ...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Dataset.mapping() -> (user_id_map, user_feature_map, item_id_map, item_feature_map)
    user_id_map, _, item_id_map, _ = dataset.mapping()

    print(f"Saving user_id_map to {USER_MAPPING_PATH} ...")
    with open(USER_MAPPING_PATH, "wb") as f:
        pickle.dump(user_id_map, f)

    print(f"Saving item_id_map to {ITEM_MAPPING_PATH} ...")
    with open(ITEM_MAPPING_PATH, "wb") as f:
        pickle.dump(item_id_map, f)

    print("Artifacts saved.")


def main():
    engine = get_engine()
    dataset, interactions, weights = build_dataset_and_interactions(engine)

    # You can tune these later
    model = train_lightfm(
        interactions,
        weights,
        no_components=64,
        epochs=20,
        num_threads=4,
    )

    save_artifacts(model, dataset)
    print("All done.")


if __name__ == "__main__":
    main()
