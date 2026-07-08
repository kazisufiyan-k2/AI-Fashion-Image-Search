import pickle
from typing import List, Tuple

import numpy as np
import torch
import open_clip
import faiss
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"   # ~350MB, well-tested, good accuracy/size tradeoff
DEVICE = "cpu"           # Render / Railway / HF Spaces free tiers are CPU-only


def load_model():
    """
    Load the open_clip model + preprocessing transform.
    Call this ONCE (wrapped in st.cache_resource in app.py) — never per-request.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    return model, preprocess


def load_index(index_path: str = "embeddings/faiss.index"):
    """Load the precomputed FAISS index from disk."""
    return faiss.read_index(index_path)


def load_image_paths(paths_pkl: str = "embeddings/image_paths.pkl") -> List[str]:
    """Load the row-aligned list of image paths that matches the FAISS index order."""
    with open(paths_pkl, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def encode_image(model, preprocess, image: Image.Image) -> np.ndarray:
    """Encode a single PIL image into a normalized (1, 512) float32 vector."""
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    feats = model.encode_image(tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def search_topk(index, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index. Returns (similarity_scores, row_indices)."""
    scores, indices = index.search(query_vector, k)
    return scores[0], indices[0]


def get_results(model, preprocess, index, image_paths: List[str], image: Image.Image, k: int = 5) -> List[dict]:
    """
    End-to-end helper used by app.py:
    query image -> encode -> FAISS search -> list of {path, similarity} dicts.
    """
    query_vec = encode_image(model, preprocess, image)
    scores, indices = search_topk(index, query_vec, k)

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1 or idx >= len(image_paths):
            continue
        results.append({
            "path": image_paths[idx],
            "similarity": float(score),
        })
    return results
