import numpy as np
import pandas as pd


def make_folds(query_ids, n_folds=5, seed=0):
    shuffled = np.random.default_rng(seed).permutation(np.asarray(query_ids))
    return pd.Series(np.arange(len(shuffled)) % n_folds, index=shuffled, name="fold").sort_index()
