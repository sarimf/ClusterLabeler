"""Offline regression tests for cluster_labeler.

Runs entirely against the built-in mock LLM (no network, no API key). Run with
``pytest`` or directly with ``python test_cluster_labeler.py``.
"""
import logging

import numpy as np
import pandas as pd

import cluster_labeler as cl
from cluster_labeler import (LabelConfig, label_clusters, labels_to_dataframe,
                             render_label_report, _coerce_fits, _as_bool)


def _toy_dataset():
    rng = np.random.default_rng(0)
    themes = {
        "billing":  ("invoice charge refund payment billing receipt", 40),
        "login":    ("password login account locked reset signin", 35),
        "shipping": ("delivery shipping tracking package arrived late", 30),
        "tiny":     ("misc oddball edge weird", 4),
    }
    centers = rng.normal(size=(len(themes), 16))
    texts, embs, cids = [], [], []
    for k, (vocab, size) in enumerate(themes.values()):
        words = vocab.split()
        for _ in range(size):
            texts.append(" ".join(rng.choice(words, size=5)))
            embs.append(centers[k] + rng.normal(scale=0.2, size=16))
            cids.append(list(themes)[k])
    df = pd.DataFrame({"text": texts, "cluster_id": cids})
    return df, np.array(embs, dtype=np.float32), themes


def test_end_to_end_mock():
    df, emb, themes = _toy_dataset()
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4), progress=False)
    assert set(cards) == set(themes)
    for cid, sc in cards.items():
        assert sc["label"]
        assert "core_texts" in sc["evidence"]          # exemplar texts are exposed
        assert sc["n_llm_calls"] >= 1
        assert sc["n_llm_calls"] < 50                   # per-cluster count, not the batch total
    tiny = cards["tiny"]
    assert tiny["scores"]["confidence"] == "unverified"
    assert "too small" in tiny["note"]
    # dataframe + report render without error
    assert len(labels_to_dataframe(cards)) == len(themes)
    assert "CLUSTER LABELS" in render_label_report(cards)


def test_fits_coercion():
    assert _coerce_fits(["true", "false", "TRUE", 1, 0], 5) == [True, False, True, True, False]
    assert _coerce_fits(None, 3) == [False, False, False]
    assert _coerce_fits([True], 3) == [True, False, False]      # padded
    assert _coerce_fits([True, True, True], 2) == [True, True]  # truncated
    assert _as_bool("false") is False and _as_bool("yes") is True


def test_input_validation():
    df, emb, _ = _toy_dataset()
    for exc, fn in [
        (KeyError, lambda: label_clusters(df.drop(columns=["text"]), embeddings=emb)),
        (ValueError, lambda: label_clusters(df, embeddings=emb[:10])),     # row mismatch
        (ValueError, lambda: label_clusters(df.iloc[:0], embeddings=emb[:0])),
    ]:
        try:
            fn()
            raise AssertionError(f"expected {exc.__name__}")
        except exc:
            pass
    bad = emb.copy(); bad[0, 0] = np.nan
    try:
        label_clusters(df, embeddings=bad)
        raise AssertionError("expected ValueError for NaN embeddings")
    except ValueError:
        pass


def test_failed_cluster_is_isolated():
    df, emb, themes = _toy_dataset()
    orig = cl._label_one_cluster

    def boom(code, cid, ctx):
        if cid == "login":
            raise RuntimeError("synthetic failure")
        return orig(code, cid, ctx)

    cl._label_one_cluster = boom
    logging.getLogger("cluster_labeler").setLevel(logging.CRITICAL)  # silence expected error
    try:
        cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4), progress=False)
    finally:
        cl._label_one_cluster = orig
        logging.getLogger("cluster_labeler").setLevel(logging.WARNING)
    assert set(cards) == set(themes)                       # batch survived
    assert cards["login"]["scores"]["confidence"] == "error"
    assert "failed" in cards["login"]["note"]


if __name__ == "__main__":
    test_end_to_end_mock()
    test_fits_coercion()
    test_input_validation()
    test_failed_cluster_is_isolated()
    print("all tests passed")
