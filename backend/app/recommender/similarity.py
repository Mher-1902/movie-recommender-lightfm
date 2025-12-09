from typing import List, Tuple, Dict

import os
import numpy as np
from sqlalchemy import create_engine, text

from .model_io import load_model_and_mappings


# DB CONFIG – will read from env when running in Docker
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlpass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "movielens")

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    return create_engine(CONN_STR)


def get_movie_titles(movie_ids: List[int]) -> Dict[int, str]:
    """
    Given a list of movie_ids, return {movie_id: title} from the movies table.
    """
    if not movie_ids:
        return {}

    engine = get_engine()
    # simple IN clause – safe here because movie_ids are ints we control
    ids_str = ", ".join(str(int(mid)) for mid in movie_ids)
    query = text(f"SELECT movie_id, title FROM movies WHERE movie_id IN ({ids_str})")

    with engine.connect() as conn:
        result = conn.execute(query)
        return {int(row.movie_id): row.title for row in result}


def get_similar_movies(
    movie_id: int,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Given a movie_id, return [(similar_movie_id, similarity_score), ...]
    using cosine similarity over LightFM item embeddings.
    """
    model, _, item_id_map = load_model_and_mappings()

    # 🔑 Normalize movie_id to int to match keys in item_id_map
    movie_id_int = int(movie_id)

    if movie_id_int not in item_id_map:
        raise ValueError(f"movie_id {movie_id_int} not found in item_id_map")

    target_idx = item_id_map[movie_id_int]

    item_embeddings = model.item_embeddings  # shape: (n_items, no_components)
    target_vec = item_embeddings[target_idx]

    # cosine similarity with all items
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    target_norm = np.linalg.norm(target_vec)
    denom = item_norms * target_norm
    denom[denom == 0.0] = 1e-10

    scores = item_embeddings @ target_vec / denom

    # sort indices by descending similarity
    ranked_indices = np.argsort(-scores)

    # reverse map: index -> raw movie_id (int)
    reverse_item_map = {v: k for k, v in item_id_map.items()}

    similar_items: List[Tuple[int, float]] = []
    for idx in ranked_indices:
        if idx == target_idx:
            continue  # skip itself
        sim_movie_id = reverse_item_map[idx]  # already int
        score = float(scores[idx])
        similar_items.append((sim_movie_id, score))
        if len(similar_items) >= top_k:
            break

    return similar_items


def get_similar_movies_with_titles(
    movie_id: int,
    top_k: int = 10,
):
    """
    Wrapper that returns a list of dicts:
    [{movie_id, title, score}, ...]
    """
    similar = get_similar_movies(movie_id, top_k=top_k)
    ids = [mid for (mid, _) in similar]
    titles = get_movie_titles(ids)

    result = []
    for mid, score in similar:
        result.append(
            {
                "movie_id": mid,
                "title": titles.get(mid),
                "score": score,
            }
        )
    return result
