from pathlib import Path
import pickle
from typing import Dict, Tuple

from lightfm import LightFM

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "lightfm_model.pkl"
USER_MAPPING_PATH = ARTIFACTS_DIR / "user_id_map.pkl"
ITEM_MAPPING_PATH = ARTIFACTS_DIR / "item_id_map.pkl"


def load_model_and_mappings() -> Tuple[LightFM, Dict[str, int], Dict[str, int]]:
    """
    Load the trained LightFM model and ID mappings from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not USER_MAPPING_PATH.exists():
        raise FileNotFoundError(f"user_id_map not found at {USER_MAPPING_PATH}")
    if not ITEM_MAPPING_PATH.exists():
        raise FileNotFoundError(f"item_id_map not found at {ITEM_MAPPING_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(USER_MAPPING_PATH, "rb") as f:
        user_id_map = pickle.load(f)

    with open(ITEM_MAPPING_PATH, "rb") as f:
        item_id_map = pickle.load(f)

    return model, user_id_map, item_id_map
