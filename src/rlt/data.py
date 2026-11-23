from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
URL = f"https://raw.githubusercontent.com/wayfair/WANDS/{COMMIT}/dataset"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FILES = ["product", "query", "label"]
GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
TEXTS = {
    "title": ["product_name"],
    "full": [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
    ],
}


def download(data_dir=DATA_DIR):
    data_dir.mkdir(exist_ok=True)
    for name in FILES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            urlretrieve(f"{URL}/{name}.csv", path)


def load(data_dir=DATA_DIR):
    download(data_dir)
    products, queries, labels = (pd.read_csv(data_dir / f"{name}.csv", sep="\t") for name in FILES)
    labels = labels.sort_values("id").drop_duplicates(["query_id", "product_id"], keep="last")
    labels["grade"] = labels["label"].map(GRADES)
    return (
        products.set_index("product_id"),
        queries.set_index("query_id"),
        labels[["query_id", "product_id", "label", "grade"]],
    )


def product_text(products, fields):
    return products[fields].fillna("").astype(str).agg(" ".join, axis=1)
