import bm25s
import numpy as np
import Stemmer


def bm25_search(texts, queries, k=1000):
    stemmer = Stemmer.Stemmer("english")
    corpus = bm25s.tokenize(list(texts), stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus, show_progress=False)
    tokens = bm25s.tokenize(list(queries), stopwords="en", stemmer=stemmer, show_progress=False)
    positions, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    return np.asarray(texts.index)[positions], scores
