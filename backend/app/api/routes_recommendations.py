from typing import List, Dict

import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from ..recommender.similarity import get_similar_movies_with_titles


router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# ---- DB CONFIG (env-based) ----
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlpass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "movielens")

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    return create_engine(CONN_STR)


# ---- RESPONSE MODEL ----

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str | None = None
    score: float


# ---- TITLE LOOKUP HELPER ----

from typing import Dict
from fastapi import HTTPException
from sqlalchemy import text

def find_movie_by_title(engine, title: str) -> Dict[str, str | int]:
    """
    Try to find a movie by title in a more human way:
    - Try several normalized variants:
        * original string
        * spaces replaced by dashes (e.g. 'Spider man' -> 'Spider-man')
    - For each variant:
        1. Exact ILIKE match
        2. Starts-with match
        3. Substring suggestions (up to 10)
    If no exact/prefix match is found but we have substring matches, we raise
    a 404 with suggestions in the payload.
    """
    cleaned = title.strip()

    # Variants to try: as-is + with spaces replaced by dashes
    variants = [cleaned]
    dashed = cleaned.replace(" ", "-")
    if dashed != cleaned:
        variants.append(dashed)

    suggestions_acc: list[Dict[str, str | int]] = []

    with engine.connect() as conn:
        for variant in variants:
            # 1) Exact ILIKE match
            exact = conn.execute(
                text("SELECT movie_id, title FROM movies WHERE title ILIKE :t LIMIT 1"),
                {"t": variant},
            ).fetchone()
            if exact:
                return {"movie_id": exact.movie_id, "title": exact.title}

            # 2) Starts-with match: 'Toy Story%' or 'Spider-Man%'
            prefix = conn.execute(
                text(
                    "SELECT movie_id, title "
                    "FROM movies "
                    "WHERE title ILIKE :t "
                    "ORDER BY movie_id "
                    "LIMIT 1"
                ),
                {"t": variant + '%'},
            ).fetchone()
            if prefix:
                return {"movie_id": prefix.movie_id, "title": prefix.title}

            # 3) Substring suggestions
            rows = conn.execute(
                text(
                    "SELECT movie_id, title "
                    "FROM movies "
                    "WHERE title ILIKE :t "
                    "ORDER BY movie_id "
                    "LIMIT 10"
                ),
                {"t": '%' + variant + '%'},
            ).fetchall()

            for row in rows:
                suggestions_acc.append(
                    {"movie_id": row.movie_id, "title": row.title}
                )

    # If we found substring matches for any variant, return them as suggestions
    if suggestions_acc:
        # Deduplicate by movie_id
        unique = {}
        for s in suggestions_acc:
            unique[s["movie_id"]] = s
        suggestions = list(unique.values())

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"No exact movie found for '{title}'.",
                "suggestions": suggestions[:10],
            },
        )

    # Nothing matched at all
    raise HTTPException(
        status_code=404,
        detail=f"No movie found matching title '{title}'",
    )



# ---- RECOMMEND BY RAW MOVIE ID (optional but handy) ----

@router.get("/by-id", response_model=List[MovieRecommendation])
def recommend_by_id(
    movie_id: int = Query(..., description="Base movie_id to recommend from"),
    top_k: int = Query(10, ge=1, le=50),
):
    """
    Given a movie_id, return top_k similar movies based on LightFM embeddings.
    """
    try:
        recommendations = get_similar_movies_with_titles(movie_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not recommendations:
        raise HTTPException(status_code=404, detail="No similar movies found.")

    return recommendations


# ---- RECOMMEND BY TITLE ----

@router.get("/by-title", response_model=List[MovieRecommendation])
def recommend_by_title(
    title: str = Query(..., description="Movie title, partial or full (e.g. 'Toy Story' or 'Spider-Man')"),
    top_k: int = Query(10, ge=1, le=50),
):
    """
    Given a (partial) movie title, find the best matching movie in the DB
    using a tolerant search, then return top_k similar movies based on
    LightFM embeddings.

    If no exact/prefix match is found but there are substring matches,
    a 404 with a list of `suggestions` is returned.
    """
    engine = get_engine()

    # This will either return a dict with movie_id/title or raise HTTPException with 404
    base_movie = find_movie_by_title(engine, title)
    base_movie_id = int(base_movie["movie_id"])

    try:
        recommendations = get_similar_movies_with_titles(base_movie_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not recommendations:
        raise HTTPException(status_code=404, detail="No similar movies found.")

    return recommendations
