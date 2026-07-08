import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import open_clip
import faiss
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cpu"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    return model, preprocess


def build(dataset_csv: str, images_dir: str, out_dir: str, batch_size: int = 32):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(dataset_csv, on_bad_lines="skip")
    if "id" not in df.columns:
        raise ValueError("styles.csv must contain an 'id' column")

    df["image_path"] = df["id"].astype(str).apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Found {len(df)} images with matching files.")

    if len(df) == 0:
        raise RuntimeError(
            f"No images matched between '{dataset_csv}' and '{images_dir}'. "
            "Check that image filenames are '<id>.jpg' and styles.csv ids match."
        )

    model, preprocess = load_model()

    all_vecs = []
    all_paths = []
    batch_imgs, batch_paths = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            tensor = torch.stack(batch_imgs)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_vecs.append(feats.cpu().numpy().astype("float32"))
        all_paths.extend(batch_paths)
        batch_imgs.clear()
        batch_paths.clear()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"skip {row['image_path']}: {e}")
            continue
        batch_imgs.append(preprocess(img))
        batch_paths.append(row["image_path"])
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    embeddings = np.concatenate(all_vecs, axis=0)
    print("Embeddings shape:", embeddings.shape)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "image_paths.pkl"), "wb") as f:
        pickle.dump(all_paths, f)

    print(f"Saved embeddings.npy, faiss.index, image_paths.pkl to '{out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/styles.csv")
    parser.add_argument("--images", default="dataset/images")
    parser.add_argument("--out", default="embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build(args.dataset, args.images, args.out, args.batch_size)
