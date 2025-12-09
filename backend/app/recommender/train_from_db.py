import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k
import pickle


# =========================
# DB CONFIG
# =========================
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlpass")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "movielens")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# =========================
# DATA LOADING
# =========================
def load_data(min_rating: float = 4.0):
    """
    Load:
    - positive interactions (user_id, movie_id) where rating >= min_rating
    - all movie_ids from the movies table (full catalog for item_id_map)
    """
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Positive interactions
        ratings_res = conn.execute(
            text(
                "SELECT user_id, movie_id, rating "
                "FROM ratings "
                "WHERE rating >= :min_rating"
            ),
            {"min_rating": min_rating},
        )
        ratings_rows = ratings_res.fetchall()

        # Full catalog of items
        movies_res = conn.execute(text("SELECT movie_id FROM movies"))
        movies_rows = movies_res.fetchall()

    if not ratings_rows:
        raise ValueError(
            f"No ratings found with rating >= {min_rating}. "
            "Try lowering min_rating."
        )

    user_ids_pos = np.array([r[0] for r in ratings_rows], dtype=np.int64)
    movie_ids_pos = np.array([r[1] for r in ratings_rows], dtype=np.int64)
    all_movie_ids = np.array([m[0] for m in movies_rows], dtype=np.int64)

    return user_ids_pos, movie_ids_pos, all_movie_ids


# =========================
# INTERACTIONS BUILDING
# =========================
def build_interactions(
    user_ids_pos: np.ndarray,
    movie_ids_pos: np.ndarray,
    all_movie_ids: np.ndarray,
):
    """
    Build implicit interactions matrix and ID mappings.

    - user_id_map: only users that have at least one positive rating
    - item_id_map: ALL movies from movies table (full catalog coverage)
    """
    # Users: only those with positives
    unique_users = np.unique(user_ids_pos)

    # Items: full catalog, not just those with positive ratings
    unique_items = np.unique(all_movie_ids)

    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

    # Map positive interactions to matrix indices
    row = np.vectorize(user_id_map.get)(user_ids_pos)
    col = np.vectorize(item_id_map.get)(movie_ids_pos)

    data = np.ones(len(user_ids_pos), dtype=np.float32)

    interactions = coo_matrix(
        (data, (row, col)),
        shape=(len(unique_users), len(unique_items)),
    )

    return interactions, user_id_map, item_id_map


# =========================
# MODEL
# =========================
def build_model() -> LightFM:
    model = LightFM(
        no_components=27,          # your choice
        loss="warp",
        learning_schedule="adagrad",
        learning_rate=0.03,
        item_alpha=1e-5,
        user_alpha=1e-5,
    )
    return model


def train_with_early_stopping(
    model: LightFM,
    train,
    test,
    max_epochs: int = 5,
    patience: int = 4,
    k_eval: int = 10,
    num_threads: int = 2,
):
    best_auc = -np.inf
    best_epoch = -1
    best_model_state = None
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.fit_partial(train, epochs=1, num_threads=num_threads)

        val_auc = auc_score(model, test, num_threads=num_threads).mean()
        val_prec = precision_at_k(
            model, test, k=k_eval, num_threads=num_threads
        ).mean()

        print(
            f"Epoch {epoch:02d} | "
            f"val AUC = {val_auc:.4f} | "
            f"val precision@{k_eval} = {val_prec:.4f}"
        )

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improve = 0
            best_model_state = {
                "item_embeddings": model.item_embeddings.copy(),
                "user_embeddings": model.user_embeddings.copy(),
                "item_biases": model.item_biases.copy(),
                "user_biases": model.user_biases.copy(),
            }
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, best AUC {best_auc:.4f})"
                )
                break

    if best_model_state is not None:
        model.item_embeddings[:] = best_model_state["item_embeddings"]
        model.user_embeddings[:] = best_model_state["user_embeddings"]
        model.item_biases[:] = best_model_state["item_biases"]
        model.user_biases[:] = best_model_state["user_biases"]

    return model, best_auc


# =========================
# SAVING
# =========================
def save_artifacts(model, user_id_map, item_id_map):
    model_path = ARTIFACTS_DIR / "lightfm_model.pkl"
    user_map_path = ARTIFACTS_DIR / "user_id_map.pkl"
    item_map_path = ARTIFACTS_DIR / "item_id_map.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(user_map_path, "wb") as f:
        pickle.dump(user_id_map, f)

    with open(item_map_path, "wb") as f:
        pickle.dump(item_id_map, f)

    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved user_id_map to {user_map_path}")
    print(f"✅ Saved item_id_map to {item_map_path}")


# =========================
# MAIN
# =========================
def main():
    print("🔹 Loading data from DB (positives + full catalog)...")
    user_ids_pos, movie_ids_pos, all_movie_ids = load_data(min_rating=4.0)
    print(f"Loaded {len(user_ids_pos)} positive interactions (rating >= 4).")

    print("🔹 Building interactions matrix...")
    interactions, user_id_map, item_id_map = build_interactions(
        user_ids_pos,
        movie_ids_pos,
        all_movie_ids,
    )
    print(
        f"Interactions matrix shape = {interactions.shape}, "
        f"nnz = {interactions.nnz}"
    )

    print("🔹 Creating train/test split...")
    train, test = random_train_test_split(
        interactions,
        test_percentage=0.2,
        random_state=42,
    )

    print("🔹 Building model...")
    model = build_model()

    print("🔹 Training with early stopping...")
    model, best_auc = train_with_early_stopping(
        model,
        train,
        test,
        max_epochs=5,
        patience=4,
        k_eval=10,
        num_threads=2,
    )
    print(f"✅ Best validation AUC: {best_auc:.4f}")

    print("🔹 Saving artifacts...")
    save_artifacts(model, user_id_map, item_id_map)
    print("✅ Done.")


if __name__ == "__main__":
    main()
