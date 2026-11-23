import bm25s
import numpy as np
import Stemmer
import torch
from sentence_transformers import SentenceTransformer

QWEN_PROMPT = "Instruct: Given a product search query, retrieve relevant products\nQuery:"
DENSE = {
    "arctic-m": ("Snowflake/snowflake-arctic-embed-m-v2.0", {"prompt_name": "query"}),
    "arctic-l": ("Snowflake/snowflake-arctic-embed-l-v2.0", {"prompt_name": "query"}),
    "bge-m3": ("BAAI/bge-m3", {}),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", {"prompt": QWEN_PROMPT}),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", {"prompt": QWEN_PROMPT}),
    "pplx-4b": ("perplexity-ai/pplx-embed-v1-4b", {}),
}


def bm25_search(texts, queries, k=1000):
    stemmer = Stemmer.Stemmer("english")
    corpus = bm25s.tokenize(list(texts), stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus, show_progress=False)
    tokens = bm25s.tokenize(list(queries), stopwords="en", stemmer=stemmer, show_progress=False)
    positions, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    return np.asarray(texts.index)[positions], scores


def rrf(runs, k=60, depth=1000):
    fused_ids, fused_scores = [], []
    for query_runs in zip(*runs, strict=True):
        totals = {}
        for ids, scores in query_runs:
            for rank, product in enumerate(ids[scores > 0], 1):
                totals[product] = totals.get(product, 0) + 1 / (k + rank)
        top = sorted(totals.items(), key=lambda item: -item[1])[:depth]
        fused_ids.append([product for product, _ in top])
        fused_scores.append([score for _, score in top])
    return np.array(fused_ids), np.array(fused_scores)


def load_dense(name, max_length=512):
    model = SentenceTransformer(
        DENSE[name][0],
        trust_remote_code=name == "pplx-4b",
        model_kwargs={"dtype": torch.bfloat16},
    )
    model.max_seq_length = max_length
    return model


def dense_search(model, name, texts, queries, k=1000, batch_size=32):
    encode = {"normalize_embeddings": True, "convert_to_tensor": True}
    products = model.encode(list(texts), batch_size=batch_size, **encode)
    query_vectors = model.encode(list(queries), **DENSE[name][1], **encode)
    scores, positions = (query_vectors @ products.T).float().topk(k, dim=1)
    return np.asarray(texts.index)[positions.cpu().numpy()], scores.cpu().numpy()
