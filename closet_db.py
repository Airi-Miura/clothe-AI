from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
CLOSET_DB_PATH = BASE_DIR / "closet_db.json"
CLOSET_IMAGE_DIR = BASE_DIR / "closet_images"


def load_closet_items() -> list[dict]:
    if not CLOSET_DB_PATH.exists():
        return []
    try:
        with CLOSET_DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def save_closet_items(items: list[dict]) -> None:
    CLOSET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    serializable_items = [_to_serializable_item(item) for item in items]
    with CLOSET_DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(serializable_items, f, ensure_ascii=False, indent=2)


def save_uploaded_closet_image(image_bytes: bytes, original_filename: str) -> str:
    CLOSET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".jpg"
    stem = _safe_stem(Path(original_filename).stem)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}_{stem}{suffix}"
    image_path = CLOSET_IMAGE_DIR / filename
    image_path.write_bytes(image_bytes)
    return str(image_path)


def delete_closet_item(items: list[dict], index: int, delete_image: bool = True) -> list[dict]:
    if not 0 <= index < len(items):
        return items
    item = items.pop(index)
    if delete_image:
        image_path = item.get("image_path")
        if image_path:
            try:
                path = Path(image_path)
                if path.exists() and path.resolve().parent == CLOSET_IMAGE_DIR.resolve():
                    path.unlink()
            except Exception:
                pass
    save_closet_items(items)
    return items


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "closet_item"


def _to_serializable_item(item: dict) -> dict:
    allowed_keys = {
        "name",
        "image_path",
        "image_name",
        "image_type",
        "category",
        "categories",
        "clo",
        "total_clo",
        "method",
        "confidence",
        "registered_at",
    }
    cleaned = {key: item.get(key) for key in allowed_keys if key in item}
    if "category" not in cleaned and item.get("categories"):
        cleaned["category"] = item["categories"][0]
    if "categories" not in cleaned and item.get("category"):
        cleaned["categories"] = [item["category"]]
    if "clo" not in cleaned and "total_clo" in cleaned:
        cleaned["clo"] = cleaned["total_clo"]
    if "total_clo" not in cleaned and "clo" in cleaned:
        cleaned["total_clo"] = cleaned["clo"]
    return cleaned
