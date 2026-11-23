import numpy as np


def gains(ranked_ids, query_ids, labels):
    grades = labels.set_index(["query_id", "product_id"])["grade"].to_dict()
    return np.array(
        [[grades.get((q, p), 0) for p in row] for q, row in zip(query_ids, ranked_ids, strict=True)]
    )


def ndcg_at(ranked_gains, query_ids, labels, k):
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = labels.groupby("query_id")["grade"].apply(lambda g: np.sort(g.to_numpy())[::-1][:k])
    idcg = np.array([ideal[q] @ discount[: len(ideal[q])] for q in query_ids])
    dcg = ranked_gains[:, :k] @ discount
    return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), np.nan)


def rcap_at(ranked_gains, query_ids, labels, k, min_grade):
    relevant = labels[labels["grade"] >= min_grade].groupby("query_id").size()
    n_relevant = relevant.reindex(query_ids, fill_value=0).to_numpy()
    found = (ranked_gains[:, :k] >= min_grade).sum(1)
    return np.where(n_relevant > 0, found / np.clip(n_relevant, 1, k), np.nan)


def paired_test(a, b, n=10_000, seed=0):
    diff = a - b
    diff = diff[~np.isnan(diff)]
    signs = np.random.default_rng(seed).choice([-1, 1], size=(n, len(diff)))
    null = np.abs((signs * diff).mean(1))
    return (np.sum(null >= abs(diff.mean())) + 1) / (n + 1)


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result
