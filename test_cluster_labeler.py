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
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4, allow_mock=True),
                           progress=False, verbose=0)
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
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True), progress=False, verbose=0)
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
        cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(workers=4, allow_mock=True),
                               progress=False, verbose=0)
    finally:
        cl._label_one_cluster = orig
        logging.getLogger("cluster_labeler").setLevel(logging.WARNING)
    assert set(cards) == set(themes)                       # batch survived
    assert cards["login"]["scores"]["confidence"] == "error"
    assert "failed" in cards["login"]["note"]


def test_report_blocks_are_grouped_under_concurrency():
    # With multiple workers, each cluster's buffered stage lines must flush as one
    # contiguous block right after its header — never interleaved with other clusters.
    import io, re, contextlib
    df, emb, themes = _toy_dataset()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True, workers=4),
                       progress=False, verbose=2)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    header = re.compile(r"^\S+ \[\d+/\d+\] \[")            # "<glyph> [i/N] [cid] ..."
    header_idx = [i for i, ln in enumerate(lines) if header.match(ln)]
    assert len(header_idx) == len(themes)                  # one header per cluster
    # the first detail line of every cluster ("· start") must immediately follow its header,
    # which only holds if the block was emitted atomically (no interleaving)
    for i in header_idx:
        assert lines[i + 1].lstrip().startswith("· start"), lines[i + 1]
    # and the returned cards must not carry the internal buffer
    assert "_log" not in label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True),
                                        progress=False, verbose=0)["billing"]


def test_call_bar_counts_llm_calls():
    # the "llm calls" tqdm bar must tick exactly once per LLM call
    if cl.tqdm is None:
        return  # tqdm not installed -> bars disabled, nothing to check
    import io, contextlib
    df, emb, _ = _toy_dataset()
    captured = {}
    orig = cl._LLMClient

    class Probe(cl._LLMClient):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["client"] = self

    cl._LLMClient = Probe
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # swallow the bar rendering
            cl.label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True, workers=4),
                              progress=True, verbose=0)
    finally:
        cl._LLMClient = orig
    c = captured["client"]
    assert c.call_bar is not None
    assert c.call_bar.n == c.n_calls > 0


def test_hanging_gateway_times_out_not_deadlocks():
    # A stalled gateway call must NOT hang the whole batch: request_timeout bounds
    # each call so label_clusters still returns (with fallback labels).
    import time
    df = pd.DataFrame({"text": ["alpha beta gamma"] * 4 + ["delta epsilon zeta"] * 4,
                       "cluster_id": ["alpha"] * 4 + ["beta"] * 4})
    emb = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32)

    def hanging_gateway(messages, json_mode=True):
        time.sleep(30)            # simulate a network request that never returns in time
        return "{}"

    cfg = LabelConfig(request_timeout=0.2, max_retries=0, workers=4)
    t0 = time.time()
    cards = label_clusters(df, embeddings=emb, llm_fn=hanging_gateway, cfg=cfg,
                           progress=False, verbose=0)
    assert time.time() - t0 < 10, "batch hung instead of timing out the stalled gateway"
    assert set(cards) == {"alpha", "beta"}      # both clusters still produced cards


def test_timeout_disabled_passes_through():
    from cluster_labeler import _call_with_timeout
    assert _call_with_timeout(lambda: 42, None) == 42
    assert _call_with_timeout(lambda: 42, 0) == 42
    assert _call_with_timeout(lambda: 42, 5) == 42
    try:
        _call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 5)
        raise AssertionError("expected the gateway error to propagate")
    except ValueError:
        pass


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
    test_report_blocks_are_grouped_under_concurrency()
    test_call_bar_counts_llm_calls()
    test_hanging_gateway_times_out_not_deadlocks()
    test_timeout_disabled_passes_through()
    print("all tests passed")
