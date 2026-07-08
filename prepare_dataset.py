"""
prepare_dataset.py — Optional helper.

If you have the full 44k styles.csv but only kept e.g. 2000 images locally,
this script filters the CSV down to just the rows whose image file actually
exists in dataset/images, and overwrites dataset/styles.csv with that subset.

Usage:
    python prepare_dataset.py --full-csv path/to/full_styles.csv --images dataset/images --out dataset/styles.csv
"""

import argparse
import os
import pandas as pd


def prepare(full_csv: str, images_dir: str, out_csv: str):
    df = pd.read_csv(full_csv, on_bad_lines="skip")
    if "id" not in df.columns:
        raise ValueError("CSV must contain an 'id' column")

    exists_mask = df["id"].astype(str).apply(lambda i: os.path.exists(os.path.join(images_dir, f"{i}.jpg")))
    filtered = df[exists_mask].reset_index(drop=True)

    filtered.to_csv(out_csv, index=False)
    print(f"Kept {len(filtered)} / {len(df)} rows that have a matching image in '{images_dir}'.")
    print(f"Wrote filtered CSV to '{out_csv}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-csv", required=True)
    parser.add_argument("--images", default="dataset/images")
    parser.add_argument("--out", default="dataset/styles.csv")
    args = parser.parse_args()
    prepare(args.full_csv, args.images, args.out)
