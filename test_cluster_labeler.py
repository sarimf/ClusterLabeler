"""Offline regression tests for cluster_labeler.

Runs entirely against the built-in mock LLM (no network, no API key). Run with
``pytest`` or directly with ``python test_cluster_labeler.py``.
"""
import logging

import numpy as np
import pandas as pd

import cluster_labeler as cl
from cluster_labeler import (LabelConfig, label_clusters, labels_to_dataframe,
                             render_label_report, _coerce_fits, _as_bool, _confidence_band)

# every test runs on the offline mock; silence the (expected) "no gateway" warning
logging.getLogger("cluster_labeler").setLevel(logging.ERROR)


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
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4, allow_mock=True), progress=False)
    assert set(cards) == set(themes)
    for cid, sc in cards.items():
        assert sc["label"]
        assert "core_texts" in sc["evidence"]          # exemplar texts are exposed
        assert sc["n_llm_calls"] >= 1
        assert sc["n_llm_calls"] < 50                   # per-cluster count, not the batch total
    tiny = cards["tiny"]
    assert tiny["scores"]["confidence"] == "unverified"
    assert "too small" in tiny["note"]
    # verified clusters expose the full, correctly-named metric set
    for cid in ("billing", "login", "shipping"):
        s = cards[cid]["scores"]
        assert set(s) >= {"recall", "precision", "specificity", "discrimination"}
    # dataframe + report render without error and carry specificity
    dfc = labels_to_dataframe(cards)
    assert len(dfc) == len(themes)
    assert "specificity" in dfc.columns
    assert "CLUSTER LABELS" in render_label_report(cards)


def test_confidence_gating_enforces_recall_and_specificity():
    cfg = LabelConfig()  # accept_discrimination=.8, accept_recall=.7, accept_precision=.7
    # high discrimination but poor recall must NOT earn "high"
    assert _confidence_band(cfg, {"recall": 0.45, "specificity": 0.95, "discrimination": 0.70}, 1.0) != "high"
    # clears every bar -> high
    assert _confidence_band(cfg, {"recall": 0.9, "specificity": 0.9, "discrimination": 0.9}, 1.0) == "high"
    # unstable label is capped below high
    assert _confidence_band(cfg, {"recall": 0.9, "specificity": 0.9, "discrimination": 0.9}, 0.0) != "high"
    # no verification at all -> unverified
    assert _confidence_band(cfg, {"recall": None, "specificity": None, "discrimination": None}, None) == "unverified"


def test_single_cluster_flags_recall_only():
    # one cluster: no siblings, so no negatives -> note must flag recall-only scoring
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"text": ["alpha beta gamma"] * 12, "cluster_id": ["only"] * 12})
    emb = rng.normal(size=(12, 8)).astype(np.float32)
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True), progress=False)
    sc = cards["only"]
    assert sc["scores"]["specificity"] is None
    assert sc["note"] and "recall-only" in sc["note"]
    # with no siblings to reject, a label cannot earn "high" confidence
    assert sc["scores"]["confidence"] != "high"


def test_parse_json_robustness():
    from cluster_labeler import _parse_json, _candidates_of, _fits_of
    # trailing prose around an object
    assert _parse_json('Sure!\n{"label": "X"}\nHope that helps') == {"label": "X"}
    # code fence
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    # bare top-level array, recovered for both candidates and fits
    assert _candidates_of(_parse_json('[{"label":"A"},{"label":"B"}]')) == [{"label": "A"}, {"label": "B"}]
    assert _fits_of(_parse_json("[true, false, true]"), 3) == [True, False, True]
    # single bare card object
    assert _candidates_of({"label": "Solo"}) == [{"label": "Solo"}]
    assert _parse_json("not json at all") is None


def test_confidence_caps_without_specificity():
    cfg = LabelConfig()
    # strong recall + discrimination but specificity never measured -> not "high"
    m = {"recall": 0.95, "precision": None, "specificity": None, "discrimination": 0.95}
    assert _confidence_band(cfg, m, None) == "medium"


def test_fits_coercion():
    assert _coerce_fits(["true", "false", "TRUE", 1, 0], 5) == [True, False, True, True, False]
    assert _coerce_fits(None, 3) == [False, False, False]
    assert _coerce_fits([True], 3) == [True, False, False]      # padded
    assert _coerce_fits([True, True, True], 2) == [True, True]  # truncated
    assert _as_bool("false") is False and _as_bool("yes") is True


def test_requires_gateway_unless_allow_mock():
    df, emb, _ = _toy_dataset()
    try:
        label_clusters(df, embeddings=emb, progress=False)  # no gateway, allow_mock defaults False
        raise AssertionError("expected ValueError when no gateway and allow_mock=False")
    except ValueError as e:
        assert "gateway" in str(e).lower()


def test_input_validation():
    df, emb, _ = _toy_dataset()
    mock = LabelConfig(allow_mock=True)  # isolate data validation from the gateway guard
    for exc, fn in [
        (KeyError, lambda: label_clusters(df.drop(columns=["text"]), embeddings=emb, cfg=mock)),
        (ValueError, lambda: label_clusters(df, embeddings=emb[:10], cfg=mock)),   # row mismatch
        (ValueError, lambda: label_clusters(df.iloc[:0], embeddings=emb[:0], cfg=mock)),
    ]:
        try:
            fn()
            raise AssertionError(f"expected {exc.__name__}")
        except exc:
            pass
    bad = emb.copy(); bad[0, 0] = np.nan
    try:
        label_clusters(df, embeddings=bad, cfg=mock)
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
        cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4, allow_mock=True), progress=False)
    finally:
        cl._label_one_cluster = orig
        logging.getLogger("cluster_labeler").setLevel(logging.WARNING)
    assert set(cards) == set(themes)                       # batch survived
    assert cards["login"]["scores"]["confidence"] == "error"
    assert "failed" in cards["login"]["note"]


if __name__ == "__main__":
    test_end_to_end_mock()
    test_fits_coercion()
    test_confidence_gating_enforces_recall_and_specificity()
    test_confidence_caps_without_specificity()
    test_single_cluster_flags_recall_only()
    test_parse_json_robustness()
    test_requires_gateway_unless_allow_mock()
    test_input_validation()
    test_failed_cluster_is_isolated()
    print("all tests passed")
