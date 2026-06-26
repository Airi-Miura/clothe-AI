from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image

FASHIONCLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
FALLBACK_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CLOTHING_CATEGORY_CANDIDATES = [
    "半袖Tシャツ",
    "長袖Tシャツ",
    "シャツ",
    "ブラウス",
    "セーター",
    "カーディガン",
    "パーカー",
    "フリース",
    "薄手ジャケット",
    "厚手ジャケット",
    "コート",
    "ダウンジャケット",
    "ジーンズ",
    "スラックス",
    "ショートパンツ",
    "スカート",
    "ワンピース",
]

CATEGORY_PROMPTS = {
    "半袖Tシャツ": "a photo of a short sleeve t-shirt",
    "長袖Tシャツ": "a photo of a long sleeve t-shirt",
    "シャツ": "a photo of a button-up shirt",
    "ブラウス": "a photo of a blouse",
    "セーター": "a photo of a sweater",
    "カーディガン": "a photo of a cardigan",
    "パーカー": "a photo of a hoodie",
    "フリース": "a photo of a fleece jacket",
    "薄手ジャケット": "a photo of a lightweight jacket",
    "厚手ジャケット": "a photo of a heavy jacket",
    "コート": "a photo of a coat",
    "ダウンジャケット": "a photo of a down jacket",
    "ジーンズ": "a photo of jeans",
    "スラックス": "a photo of slacks trousers",
    "ショートパンツ": "a photo of shorts",
    "スカート": "a photo of a skirt",
    "ワンピース": "a photo of a dress",
}


def load_clip_classifier(preferred_model_name: str = FASHIONCLIP_MODEL_NAME) -> dict[str, Any]:
    try:
        return _load_clip_model(preferred_model_name, backend="fashionclip")
    except Exception as fashionclip_error:
        try:
            bundle = _load_clip_model(FALLBACK_CLIP_MODEL_NAME, backend="clip")
            bundle["warning"] = (
                f"FashionCLIPモデルの読み込みに失敗したため、CLIPで代替します: {fashionclip_error}"
            )
            return bundle
        except Exception as clip_error:
            raise RuntimeError(
                "FashionCLIP/CLIPモデルを読み込めませんでした。"
                "transformers, torch, Pillow のインストールとモデルキャッシュを確認してください。"
                f" FashionCLIP error: {fashionclip_error}; CLIP error: {clip_error}"
            ) from clip_error


def _load_clip_model(model_name: str, backend: str) -> dict[str, Any]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_name": model_name,
        "backend": backend,
    }


def classify_clothing_image(
    image_bytes: bytes,
    model_bundle: dict[str, Any],
    categories: list[str] | None = None,
    clo_dict: dict[str, float] | None = None,
) -> dict[str, Any]:
    import torch

    if categories is None:
        categories = CLOTHING_CATEGORY_CANDIDATES
    if clo_dict is None:
        clo_dict = {}

    model = model_bundle["model"]
    processor = model_bundle["processor"]
    device = model_bundle["device"]

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    prompts = [CATEGORY_PROMPTS.get(category, f"a photo of {category}") for category in categories]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().tolist()

    ranked = sorted(zip(categories, probs), key=lambda item: item[1], reverse=True)
    top_category, top_score = ranked[0]
    top_candidates = [
        {"category": category, "score": round(float(score), 4)}
        for category, score in ranked[:3]
    ]

    return {
        "category": top_category,
        "clo": float(clo_dict.get(top_category, 0.0)),
        "confidence": round(float(top_score), 4),
        "top_candidates": top_candidates,
        "method": "fashionclip",
        "backend": model_bundle.get("backend", "fashionclip"),
        "model_name": model_bundle.get("model_name", ""),
    }