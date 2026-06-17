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


def test_verify_positives_capped_on_large_clusters():
    # a large cluster must NOT send its whole holdout (~30% of the cluster) into a
    # verify prompt — that's what makes big clusters time out. Cap at verify_positives.
    rng = np.random.default_rng(0)
    n_big, n_other = 300, 60
    df = pd.DataFrame({
        "text": [f"alpha beta gamma item{i}" for i in range(n_big)] +
                [f"delta epsilon zeta item{i}" for i in range(n_other)],
        "cluster_id": ["big"] * n_big + ["other"] * n_other,
    })
    emb = np.vstack([
        rng.normal(0, 0.1, (n_big, 8)) + np.array([5.] + [0.] * 7),
        rng.normal(0, 0.1, (n_other, 8)) + np.array([-5.] + [0.] * 7),
    ]).astype(np.float32)

    seen = {"max_pos": 0}
    orig = cl._grade_candidate

    def spy(ctx, cand, pos_texts, neg_texts, rng):
        seen["max_pos"] = max(seen["max_pos"], len(pos_texts))
        return orig(ctx, cand, pos_texts, neg_texts, rng)

    cl._grade_candidate = spy
    try:
        cl.label_clusters(df, embeddings=emb,
                          cfg=LabelConfig(allow_mock=True, verify_positives=12, workers=2),
                          progress=False, verbose=0)
    finally:
        cl._grade_candidate = orig
    assert 0 < seen["max_pos"] <= 12, f"verify prompt used {seen['max_pos']} positives (cap 12)"


def test_use_llm_is_decorator_friendly():
    # use_llm / use_genai must return the function so @use_llm doesn't rebind the
    # decorated name to None, and must register it as the gateway.
    from cluster_labeler import use_llm, use_genai, _GATEWAY
    saved = _GATEWAY[0]
    try:
        @use_llm
        def gw(messages, json_mode=True):
            return "{}"
        assert gw is not None and callable(gw)        # name not clobbered to None
        assert gw(messages=[], json_mode=True) == "{}"  # still the real function
        assert _GATEWAY[0] is gw                        # registered as the gateway
        assert use_llm(gw) is gw and use_genai(gw) is gw
    finally:
        _GATEWAY[0] = saved


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
        assert "breadth" in sc                          # axis decomposition always present
        assert set(sc["breadth"]) >= {"invariant_summary", "varying_summary", "invariant_axes",
                                      "varying_axes", "coherence", "n_invariant", "n_varying"}
        assert sc["n_llm_calls"] >= 1
        assert sc["n_llm_calls"] < 60                   # per-cluster count, not the batch total
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


def test_llm_call_count_shown_in_progress_bar():
    # the running LLM-call count rides in the single progress bar's postfix
    # (no fragile second bar that spams lines in notebooks)
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
    assert c.n_calls > 0
    assert c.progress_bar is not None
    assert "llm calls" in (c.progress_bar.postfix or "")


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


def test_timeout_is_terminal_no_retry_storm():
    # a timed-out call must NOT be retried (the call is still running; retrying
    # spawns duplicates and, with a retrying gateway, a storm). The gateway
    # should be invoked exactly once despite max_retries=3.
    import time
    from cluster_labeler import _LLMClient
    calls = []

    def hanging(messages, json_mode=True):
        calls.append(1)
        time.sleep(30)
        return "{}"

    client = _LLMClient(LabelConfig(request_timeout=0.2, max_retries=3), hanging)
    t0 = time.time()
    res = client.complete("p", "propose", {})
    assert res == {}
    assert time.time() - t0 < 5            # ~one 0.2s timeout, not 4 × (0.2 + backoff)
    assert len(calls) == 1                  # terminal: invoked once, no storm


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


def _shared_token_dataset(shared="alpha", n_per=40):
    # every member of cluster A contains `shared` plus varying words, so the mock's
    # decompose finds an invariant axis (=> coherence is exercised).
    rng = np.random.default_rng(3)
    varyA = "laptop monitor warranty dock keyboard mouse".split()
    centers = rng.normal(size=(2, 12))
    texts, embs, cids = [], [], []
    for _ in range(n_per):
        texts.append(shared + " " + " ".join(rng.choice(varyA, 3)))
        embs.append(centers[0] + rng.normal(scale=0.2, size=12)); cids.append("A")
    for _ in range(30):
        texts.append("beta " + " ".join(rng.choice(["login", "reset", "password"], 3)))
        embs.append(centers[1] + rng.normal(scale=0.2, size=12)); cids.append("B")
    df = pd.DataFrame({"text": texts, "cluster_id": cids})
    return df, np.array(embs, dtype=np.float32)


def test_breadth_invariant_axes_and_coherence():
    df, emb = _shared_token_dataset()
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True), progress=False, verbose=0)
    a = cards["A"]["breadth"]
    assert a["n_invariant"] >= 1                         # mock finds the shared "alpha" token
    assert a["invariant_axes"][0]["value"] == "alpha"
    assert a["coherence"] is not None and 0.0 <= a["coherence"] <= 1.0
    assert a["coherence"] >= 0.9                         # held-out members all contain "alpha"
    # varying axis values are present and stringified
    assert a["n_varying"] >= 1 and all(isinstance(x, str) for x in a["varying_axes"][0]["values"])


def test_breadth_small_cluster_has_no_coherence():
    df, emb, _ = _toy_dataset()
    sc = label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True),
                        progress=False, verbose=0)["tiny"]
    assert sc["breadth"]["coherence"] is None            # no holdout on a size-4 cluster
    assert set(sc["breadth"]) >= {"invariant_summary", "varying_summary", "invariant_axes", "varying_axes"}


def test_breadth_varying_axes_capped():
    import json
    df, emb, _ = _toy_dataset()

    def gw(messages, json_mode=True):
        p = messages[-1]["content"]
        if "Decompose the TARGET" in p:
            return json.dumps({"summary": "s", "invariant_axes": [{"axis": "k", "value": "v"}],
                               "varying_axes": [{"axis": f"ax{i}", "values": ["a", "b"],
                                                 "open_ended": False} for i in range(20)]})
        if '"fits"' in p:
            return json.dumps({"fits": [True] * 64})
        if '"candidates"' in p:
            return json.dumps({"candidates": [{"label": "L", "description": "D", "rationale": "R"}]})
        return "{}"

    cards = label_clusters(df, embeddings=emb, llm_fn=gw,
                           cfg=LabelConfig(breadth_max_axes=5, max_retries=0, request_timeout=0),
                           progress=False, verbose=0)
    for sc in cards.values():
        assert sc["breadth"]["n_varying"] <= 5


def test_propose_prompt_includes_axes_guidance():
    from cluster_labeler import _propose_prompt
    p = _propose_prompt(LabelConfig(), ["t1"], ["n1"], 2,
                        [{"axis": "objection", "value": "pricing"}],
                        [{"axis": "product", "values": ["laptop", "monitor"]}])
    assert "SHARED IDENTITY" in p and "objection=pricing" in p
    assert "INCIDENTAL VARIATION" in p and "product" in p
    # additive: with no axes, the guidance block is absent
    assert "SHARED IDENTITY" not in _propose_prompt(LabelConfig(), ["t1"], ["n1"], 2)


def test_union_axes_dedupes_and_merges():
    from cluster_labeler import _union_axes
    a = [{"axis": "Product", "values": ["laptop"], "open_ended": False}]
    b = [{"axis": "product", "values": ["monitor"], "open_ended": True},
         {"axis": "tone", "values": ["angry"]}]
    out = _union_axes(a, b, merge_values=True)
    names = [x["axis"] for x in out]
    assert names == ["Product", "tone"]                 # deduped by lowercased name, order kept
    assert out[0]["values"] == ["laptop", "monitor"] and out[0]["open_ended"] is True


def test_breadth_deterministic():
    df, emb = _shared_token_dataset()
    def run():
        return label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True, workers=3),
                              progress=False, verbose=0)
    a, b = run(), run()
    for cid in a:
        ba, bb = a[cid]["breadth"], b[cid]["breadth"]
        assert (ba["n_invariant"], ba["n_varying"], ba["invariant_summary"], ba["varying_summary"]) == \
               (bb["n_invariant"], bb["n_varying"], bb["invariant_summary"], bb["varying_summary"])
        assert [v["values"] for v in ba["varying_axes"]] == [v["values"] for v in bb["varying_axes"]]


def test_breadth_collations_are_descriptive():
    from cluster_labeler import _describe_invariant, _describe_varying
    inv = _describe_invariant([{"axis": "objection", "value": "pricing"},
                               {"axis": "sentiment", "value": "negative"}])
    assert inv == "All members share objection (pricing), sentiment (negative)."
    var = _describe_varying([{"axis": "product", "values": ["laptop", "monitor"], "open_ended": True},
                             {"axis": "tone", "values": ["polite", "frustrated"]}])
    assert var == "Members vary by product (laptop, monitor, …); tone (polite, frustrated)."
    assert _describe_invariant([]) == "" and _describe_varying([]) == ""


def test_breadth_collations_in_card_and_df():
    df, emb = _shared_token_dataset()
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig(allow_mock=True), progress=False, verbose=0)
    a = cards["A"]["breadth"]
    assert a["invariant_summary"].startswith("All members share") and "alpha" in a["invariant_summary"]
    assert a["varying_summary"].startswith("Members vary by")
    dfc = labels_to_dataframe(cards).set_index("cluster_id")
    for c in ("invariant_summary", "varying_summary", "n_invariant_axes", "n_varying_axes", "coherence"):
        assert c in dfc.columns
    assert "breadth_summary" not in dfc.columns and "varying_axes" not in dfc.columns
    assert dfc.loc["A", "invariant_summary"] == a["invariant_summary"]
    rep = render_label_report(cards)
    assert "shared:" in rep and "varies:" in rep


def test_breadth_prose_uses_two_focused_asks():
    # with breadth_prose=True the summaries come from two dedicated LLM calls.
    import json
    df, emb = _shared_token_dataset()
    seen = {"inv": 0, "var": 0}

    def gw(messages, json_mode=True):
        p = messages[-1]["content"]
        if "what EVERY member of this cluster shares" in p:
            seen["inv"] += 1
            return json.dumps({"summary": "SHARED PROSE about pricing."})
        if "range of variation across members" in p:
            seen["var"] += 1
            return json.dumps({"summary": "VARY PROSE about products."})
        if "Decompose the TARGET" in p:
            return json.dumps({"invariant_axes": [{"axis": "objection", "value": "pricing"}],
                               "varying_axes": [{"axis": "product", "values": ["a", "b"], "open_ended": True}]})
        if '"fits"' in p:
            return json.dumps({"fits": [True] * 64})
        if '"candidates"' in p:
            return json.dumps({"candidates": [{"label": "L", "description": "D", "rationale": "R"}]})
        return "{}"

    cards = label_clusters(df, embeddings=emb, llm_fn=gw,
                           cfg=LabelConfig(breadth_prose=True, max_retries=0, request_timeout=0),
                           progress=False, verbose=0)
    a = cards["A"]["breadth"]
    assert a["invariant_summary"] == "SHARED PROSE about pricing."
    assert a["varying_summary"] == "VARY PROSE about products."
    assert seen["inv"] >= 1 and seen["var"] >= 1            # two focused asks happened


def test_thorough_preset():
    from cluster_labeler import LabelConfig
    c = LabelConfig.thorough()
    # the discovery knobs are turned up vs the defaults
    d = LabelConfig()
    assert c.breadth_gap_passes > d.breadth_gap_passes and c.breadth_resamples > d.breadth_resamples
    assert c.breadth_exemplars > d.breadth_exemplars and c.breadth_max_axes > d.breadth_max_axes
    assert c.micro_k >= c.n_diverse                       # diverse sample isn't silently capped
    # overrides win, and unrelated defaults are preserved
    o = LabelConfig.thorough(domain_hint="objections", breadth_gap_passes=5)
    assert o.domain_hint == "objections" and o.breadth_gap_passes == 5 and o.allow_mock is False
    # and it actually runs end-to-end on the mock
    df, emb = _shared_token_dataset()
    cards = label_clusters(df, embeddings=emb, cfg=LabelConfig.thorough(allow_mock=True),
                           progress=False, verbose=0)
    assert all("breadth" in sc for sc in cards.values())


def test_breadth_gap_pass_recovers_missing_axis():
    # the gap pass shows axes-so-far + a fresh sample and asks for what's MISSING;
    # a varying axis returned only by the gap call must end up in the breadth.
    import json
    df, emb = _shared_token_dataset()
    seen = {"gap": 0}

    def gw(messages, json_mode=True):
        p = messages[-1]["content"]
        if "axes MISSING from the lists above" in p:          # the gap prompt
            seen["gap"] += 1
            return json.dumps({"invariant_axes": [],
                               "varying_axes": [{"axis": "urgency", "values": ["high", "low"],
                                                 "open_ended": False}]})
        if "Decompose the TARGET" in p:
            return json.dumps({"invariant_axes": [{"axis": "objection", "value": "pricing"}],
                               "varying_axes": [{"axis": "product", "values": ["a"], "open_ended": True}]})
        if "EVERY member of this cluster shares" in p:
            return json.dumps({"summary": "shared"})
        if "range of variation across members" in p:
            return json.dumps({"summary": "varies"})
        if '"fits"' in p:
            return json.dumps({"fits": [True] * 64})
        if '"candidates"' in p:
            return json.dumps({"candidates": [{"label": "L", "description": "D", "rationale": "R"}]})
        return "{}"

    cards = label_clusters(df, embeddings=emb, llm_fn=gw,
                           cfg=LabelConfig(breadth_gap_passes=1, max_retries=0, request_timeout=0),
                           progress=False, verbose=0)
    axes = {a["axis"] for a in cards["A"]["breadth"]["varying_axes"]}
    assert seen["gap"] >= 1                                   # gap pass ran
    assert "urgency" in axes and "product" in axes           # missing axis recovered + original kept


def test_breadth_gap_passes_zero_disables():
    import json
    df, emb = _shared_token_dataset()

    def gw(messages, json_mode=True):
        p = messages[-1]["content"]
        assert "axes MISSING from the lists above" not in p   # gap pass must not run
        if "Decompose the TARGET" in p:
            return json.dumps({"invariant_axes": [{"axis": "objection", "value": "pricing"}],
                               "varying_axes": [{"axis": "product", "values": ["a"], "open_ended": True}]})
        if "shares" in p or "variation" in p:
            return json.dumps({"summary": "s"})
        if '"fits"' in p:
            return json.dumps({"fits": [True] * 64})
        if '"candidates"' in p:
            return json.dumps({"candidates": [{"label": "L", "description": "D", "rationale": "R"}]})
        return "{}"

    label_clusters(df, embeddings=emb, llm_fn=gw,
                   cfg=LabelConfig(breadth_gap_passes=0, max_retries=0, request_timeout=0),
                   progress=False, verbose=0)


def test_breadth_prose_off_uses_deterministic_collation_no_extra_calls():
    # breadth_prose=False must NOT issue the summary asks; falls back to collation.
    import json
    df, emb = _shared_token_dataset()

    def gw(messages, json_mode=True):
        p = messages[-1]["content"]
        assert "what EVERY member of this cluster shares" not in p   # no prose ask
        assert "range of variation across members" not in p
        if "Decompose the TARGET" in p:
            return json.dumps({"invariant_axes": [{"axis": "objection", "value": "pricing"}],
                               "varying_axes": [{"axis": "product", "values": ["a"], "open_ended": True}]})
        if '"fits"' in p:
            return json.dumps({"fits": [True] * 64})
        if '"candidates"' in p:
            return json.dumps({"candidates": [{"label": "L", "description": "D", "rationale": "R"}]})
        return "{}"

    cards = label_clusters(df, embeddings=emb, llm_fn=gw,
                           cfg=LabelConfig(breadth_prose=False, max_retries=0, request_timeout=0),
                           progress=False, verbose=0)
    a = cards["A"]["breadth"]
    assert a["invariant_summary"] == "All members share objection (pricing)."
    assert a["varying_summary"].startswith("Members vary by product")


if __name__ == "__main__":
    test_use_llm_is_decorator_friendly()
    test_breadth_invariant_axes_and_coherence()
    test_breadth_small_cluster_has_no_coherence()
    test_breadth_varying_axes_capped()
    test_propose_prompt_includes_axes_guidance()
    test_union_axes_dedupes_and_merges()
    test_breadth_deterministic()
    test_breadth_collations_are_descriptive()
    test_breadth_collations_in_card_and_df()
    test_breadth_prose_uses_two_focused_asks()
    test_thorough_preset()
    test_breadth_gap_pass_recovers_missing_axis()
    test_breadth_gap_passes_zero_disables()
    test_breadth_prose_off_uses_deterministic_collation_no_extra_calls()
    test_verify_positives_capped_on_large_clusters()
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
    test_llm_call_count_shown_in_progress_bar()
    test_hanging_gateway_times_out_not_deadlocks()
    test_timeout_is_terminal_no_retry_storm()
    test_timeout_disabled_passes_through()
    print("all tests passed")
