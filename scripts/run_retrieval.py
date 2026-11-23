import sys
import time
from pathlib import Path

import numpy as np
import torch

from rlt.data import TEXTS, load, product_text
from rlt.retrieve import DENSE, bm25_search, dense_search, load_dense

RUNS = Path(__file__).resolve().parents[1] / "runs"


def save(name, ids, scores, seconds):
    np.savez(RUNS / f"{name}.npz", ids=ids, scores=scores, query_ids=queries.index, seconds=seconds)
    print(f"{name}: {seconds:.0f} s", flush=True)


RUNS.mkdir(exist_ok=True)
products, queries, labels = load()
names = sys.argv[1:] or ["bm25", *DENSE]

for name in names:
    todo = {t: cols for t, cols in TEXTS.items() if not (RUNS / f"{name}_{t}.npz").exists()}
    if not todo:
        continue
    model = None if name == "bm25" else load_dense(name)
    for text, columns in todo.items():
        start = time.time()
        texts = product_text(products, columns)
        if model is None:
            ids, scores = bm25_search(texts, queries["query"])
        else:
            ids, scores = dense_search(model, name, texts, queries["query"])
        save(f"{name}_{text}", ids, scores, time.time() - start)
    del model
    torch.cuda.empty_cache()
