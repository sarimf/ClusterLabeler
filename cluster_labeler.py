"""cluster_labeler.py — contrastive, verified semantic labeling for text clusters.

Standalone module: no dependency on cluster_judge.py. Takes a DataFrame (text +
cluster id [+ optional embeddings]) and an LLM callable, returns a per-cluster
scorecard: label, description, rationale, confidence scores, and evidence.

Why this is not "summarize the centroid neighbours":
  A label is a decision boundary, not a caption. It is generated and graded the
  way a classifier would be:
    1. EVIDENCE   - core (typical) + diverse (sub-modes) + boundary (where this
                    cluster blurs into its nearest neighbour) + contrast samples
                    from neighbouring clusters (what this is NOT).
    2. PROPOSE    - several candidate cards, each required to be true of the
                    target and false of the neighbours (contrastive, not just
                    descriptive).
    3. VERIFY     - score each candidate as a classifier on HELD-OUT items never
                    shown during evidence/proposal: recall (held-out members
                    accepted), specificity (held-out sibling items rejected),
                    precision (of items it accepts, share that are true members),
                    discrimination (balanced accuracy of recall + specificity).
    4. REFINE     - feed the best candidate's false negatives/positives back and
                    ask for a revision; repeat until it clears a bar or the
                    iteration budget runs out.
    5. STABILITY  - resample the evidence and re-propose; a label that survives
                    resampling is trustworthy, one that flips means the cluster
                    itself is ill-defined.
    6. SUB-THEMES + GLOBAL COHERENCE - flag clusters whose sub-modes look like
                    two different things, and flag pairs of clusters whose
                    labels collide despite distinct content (re-differentiated)
                    or whose content overlaps despite distinct labels (flagged
                    for merge review).

Every claim a label makes should be backed by evidence a human can re-check;
every score should come from held-out items the candidate never saw.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    # tqdm.auto picks the notebook-native widget bar inside Jupyter and the
    # console bar elsewhere — avoids the multi-line spam plain tqdm produces in
    # notebooks (which don't support the ANSI cursor moves stacked bars need).
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

log = logging.getLogger("cluster_labeler")

__all__ = ["LabelConfig", "use_llm", "use_genai", "label_clusters", "labels_to_dataframe",
           "render_label_report"]


class _Reporter:
    """Thread-safe progress emitter. Prints to stderr (or via tqdm so a live
    bar is not corrupted) regardless of the host's logging configuration, which
    is why the old log.info() calls were invisible by default. Also mirrors
    every message to the 'cluster_labeler' logger for apps that capture logs.

    verbose levels:
      0  silent
      1  banner, one line per finished cluster, coherence flags, final summary
      2  also per-stage detail within each cluster (propose / verify / refine / …)
    """

    def __init__(self, verbose: int, use_bar: bool):
        self.verbose = int(verbose)
        self.use_bar = use_bar and tqdm is not None
        self._lock = threading.Lock()

    def __call__(self, msg: str, level: int = 1) -> None:
        log.info(msg)
        if self.verbose < level:
            return
        with self._lock:
            if self.use_bar:
                tqdm.write(msg)
            else:
                print(msg, file=sys.stderr, flush=True)


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class LabelConfig:
    domain_hint: Optional[str] = None        # e.g. "customer support chat messages"
    same_when: Optional[str] = None          # e.g. "they describe the same underlying issue"
    item_chars: int = 400
    desc_chars: int = 160
    breadth_summary_chars: int = 600         # max chars for each breadth collation
                                             # (invariant_summary / varying_summary).

    # evidence shape
    n_core: int = 8                          # nearest-centroid exemplars (the "typical" member)
    n_diverse: int = 6                       # micro-mode medoids (the spread / sub-themes)
    n_boundary: int = 4                      # items closest to the nearest sibling centroid
    micro_k: int = 6                         # micro-modes for the diverse sample
    n_contrast_clusters: int = 3             # nearest sibling clusters shown as "what this is NOT"
    n_contrast_items: int = 4                # exemplars per contrast cluster

    # held-out verification (classifier-style grading, never seen by evidence/proposal)
    holdout_frac: float = 0.3
    min_holdout: int = 4
    verify_positives: int = 12               # held-out members sampled into a verify prompt.
                                             # Caps prompt size on big clusters — without this a
                                             # large cluster sends thousands of items per call.
    verify_negatives: int = 8                # held-out sibling items used as negatives

    # candidate generation / refinement
    n_candidates: int = 4
    refine_max_iters: int = 2
    accept_discrimination: float = 0.80
    accept_recall: float = 0.70
    accept_precision: float = 0.70

    # stability (resample evidence, re-propose, compare)
    stability_resamples: int = 2
    stability_min_jaccard: float = 0.34

    # sub-theme detection (informational; never overrides the main label)
    subtheme_spread_ratio: float = 1.6       # inter/intra micro-mode spread trigger
    subtheme_min_size: int = 16

    # breadth: decompose each cluster into invariant axes (the shared identity the
    # label should name) + varying axes (the spread), computed BEFORE the label and
    # used to guide it. The invariants are then verified on held-out members.
    breadth_exemplars: int = 14              # diverse target exemplars shown to the decomposer
    breadth_max_axes: int = 8                # cap on varying axes listed
    breadth_resamples: int = 1               # >1 unions independent extractions (higher axis recall)
    breadth_gap_passes: int = 1              # after the first extraction, show the axes found so far +
                                             # a fresh extreme-weighted sample and ask ONLY for what's
                                             # MISSING; union it. Repeats toward saturation; 0 disables.
    breadth_verify: bool = True              # verify invariant axes on held-out -> coherence
    breadth_prose: bool = True               # write invariant/varying summaries as LLM prose via two
                                             # focused asks (+2 calls/cluster). False -> deterministic
                                             # collation of the axes (no extra calls).

    # global coherence pass across all clusters' finished cards
    dedup_label_jaccard: float = 0.6         # near-identical label text -> re-differentiate
    dedup_cent_sim: float = 0.55             # near-identical content -> flag for merge review

    # small clusters: skip holdout split, label from all members, mark low confidence
    min_cluster_size: int = 6

    # judge
    model: str = "mock"
    temperature: float = 0.2
    seed: int = 7
    workers: int = 16
    max_retries: int = 3
    backoff_base: float = 0.5
    request_timeout: float = 60.0            # seconds per LLM call; 0/None disables.
                                             # Bounds a stalled gateway so one hung
                                             # network request can't hang the whole batch.

    # offline mock labeler is OPT-IN: label_clusters refuses to run without a
    # registered gateway / llm_fn unless this is explicitly set true. Prevents a
    # misconfigured production run from silently emitting mock-quality labels.
    allow_mock: bool = False


_GATEWAY: List[Optional[Callable]] = [None]


def use_llm(fn: Callable[[List[dict], bool], str]) -> Callable[[List[dict], bool], str]:
    """Register the judge gateway: fn(messages, json_mode=True) -> str (raw model text).

    Returns fn unchanged so it can be used directly (use_llm(fn)) OR as a
    decorator (@use_llm) without the decorated name becoming None.
    """
    _GATEWAY[0] = fn
    return fn


# alias for naming parity with the sibling cluster_judge module (use_genai); same signature.
use_genai = use_llm


# ===========================================================================
# small helpers
# ===========================================================================
def _normalize(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (emb / n).astype(np.float32)


def _clip(t: Any, n: int) -> str:
    t = str(t).replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


_TRUE_STRS = {"true", "yes", "y", "t", "1", "fit", "fits", "member"}


def _as_bool(x: Any) -> bool:
    """Coerce a model-supplied verdict to bool. Models often return strings
    ("true"/"false") or 0/1 instead of JSON booleans; a naive ``if x`` would
    treat the string "false" as truthy and silently corrupt every score."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in _TRUE_STRS
    return bool(x)


def _coerce_fits(raw: Any, n: int) -> List[bool]:
    """Normalise a model 'fits' array to exactly n booleans (pad with False)."""
    seq = raw if isinstance(raw, list) else []
    fits = [_as_bool(v) for v in seq]
    if len(fits) < n:
        fits = fits + [False] * (n - len(fits))
    return fits[:n]


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
         "were", "be", "this", "that", "it", "with", "as", "at", "by", "from", "i", "you"}


def _tokens(s: str) -> set:
    return {w for w in _WORD_RE.findall(str(s).lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0


_OBJ_RE = re.compile(r"\{.*\}", re.S)
_ARR_RE = re.compile(r"\[.*\]", re.S)


def _parse_json(s: Any) -> Optional[Any]:
    """Parse a model reply into a dict or list. Tolerates code fences, leading/
    trailing prose, and a bare top-level array (some models drop the wrapper)."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # fall back to extracting the outermost object, then the outermost array
    for rx in (_OBJ_RE, _ARR_RE):
        m = rx.search(s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def _candidates_of(res: Any) -> List[dict]:
    """Extract candidate cards whether the model wrapped them ({"candidates":[...]}),
    returned a bare list, or returned a single card object."""
    if isinstance(res, list):
        return [c for c in res if isinstance(c, dict)]
    if isinstance(res, dict):
        cands = res.get("candidates")
        if isinstance(cands, list):
            return [c for c in cands if isinstance(c, dict)]
        if res.get("label"):
            return [res]
    return []


def _fits_of(res: Any, n: int) -> List[bool]:
    """Extract per-item verdicts whether wrapped ({"fits":[...]}) or a bare list."""
    raw = res if isinstance(res, list) else (res.get("fits") if isinstance(res, dict) else None)
    return _coerce_fits(raw, n)


def _call_with_timeout(fn: Callable[[], Any], timeout: Optional[float]) -> Any:
    """Run fn() but stop waiting after `timeout` seconds. A user gateway is an
    arbitrary (usually blocking network) call; without a bound, one stalled
    request hangs its worker forever and the whole batch never completes. The
    call runs on a daemon thread, so on timeout we abandon it (it cannot block
    interpreter exit) and raise TimeoutError, which the retry loop handles."""
    if not timeout or timeout <= 0:
        return fn()
    box: Dict[str, Any] = {}

    def _run() -> None:
        try:
            box["v"] = fn()
        except BaseException as e:  # propagate the gateway's own error to the caller
            box["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout:g}s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


# ===========================================================================
# LLM client (mock fallback = cheap extractive contrast, no network needed)
# ===========================================================================
class _LLMClient:
    def __init__(self, cfg: LabelConfig, llm_fn: Optional[Callable]):
        self.cfg = cfg
        self.fn = llm_fn
        self.mock = llm_fn is None
        if self.mock:
            # the mock is an extractive word-overlap heuristic, not a real labeler.
            # Falling back to it silently is a production footgun, so say so loudly.
            log.warning("cluster_labeler: NO LLM gateway registered — using the offline MOCK "
                        "labeler (word-overlap heuristic, NOT model quality). Register a real "
                        "model via use_llm()/use_genai() or pass llm_fn= for real labels.")
        self.n_calls = 0
        self.n_empty = 0
        self._lock = threading.Lock()
        # live LLM-call count shown in the (single) progress bar's postfix.
        # A second stacked bar spams new lines in notebooks, so we keep one bar
        # and update its suffix instead. Set by label_clusters.
        self.progress_bar = None
        self.progress_lock: Optional[threading.Lock] = None
        self._t0 = time.time()
        self._last_post = 0.0
        # per-thread call counter: each cluster is labeled start-to-finish on a
        # single worker thread, so a thread-local count is the only correct way
        # to attribute LLM calls to a cluster when workers run concurrently.
        self._local = threading.local()

    def _tick_bar(self, force: bool = False) -> None:
        """Refresh the progress bar's postfix with the running LLM-call count and
        rate. Throttled, and serialized with the main thread's bar.update() via a
        shared lock so concurrent worker updates don't corrupt the bar."""
        bar = self.progress_bar
        if bar is None:
            return
        now = time.time()
        if not force and now - self._last_post < 0.2:
            return
        self._last_post = now
        rate = self.n_calls / max(now - self._t0, 1e-9)
        text = f"{self.n_calls} llm calls · {rate:.1f}/s"
        lock = self.progress_lock
        if lock is not None:
            with lock:
                bar.set_postfix_str(text, refresh=True)
        else:
            bar.set_postfix_str(text, refresh=True)

    def reset_call_counter(self) -> None:
        self._local.count = 0

    def calls_since_reset(self) -> int:
        return getattr(self._local, "count", 0)

    def complete(self, prompt: str, mock_kind: str, mock_ctx: dict) -> dict:
        with self._lock:
            self.n_calls += 1
        self._tick_bar()                     # live count of LLM calls in the bar postfix
        self._local.count = getattr(self._local, "count", 0) + 1
        if self.mock:
            return _mock_response(mock_kind, mock_ctx)
        msgs = [{"role": "system", "content": "You are a careful analyst. Reply with STRICT JSON only."},
                {"role": "user", "content": prompt}]
        delay = self.cfg.backoff_base
        attempts = self.cfg.max_retries + 1
        for att in range(attempts):
            try:
                raw = _call_with_timeout(lambda: self.fn(msgs, json_mode=True),
                                         self.cfg.request_timeout)
                v = _parse_json(raw)
                if v:
                    return v
                # parse/empty failures used to be swallowed silently
                log.warning("LLM returned unparseable/empty JSON (%s), attempt %d/%d",
                            mock_kind, att + 1, attempts)
            except TimeoutError:
                # A timeout means the call is STILL RUNNING in the background.
                # Retrying would spawn a duplicate, and if the gateway does its
                # own retries/backoff (e.g. tenacity) the duplicates multiply
                # into a storm. So a timeout is terminal for this call.
                log.warning("LLM call timed out (%s) after %ss — giving up this call. If your "
                            "gateway already retries/backs off, set request_timeout=0 and "
                            "max_retries=0 so the two layers don't fight.",
                            mock_kind, self.cfg.request_timeout)
                break
            except Exception as e:
                log.warning("LLM call error (%s), attempt %d/%d: %s", mock_kind, att + 1, attempts, e)
            if att < attempts - 1:        # don't sleep after the final attempt
                time.sleep(delay)
                delay *= 2
        with self._lock:
            self.n_empty += 1
        return {}


def _mock_response(kind: str, ctx: dict) -> dict:
    """Offline fallback: simple word-contrast heuristic so the pipeline is testable
    without a real model registered. Not meant to produce good labels."""
    if kind == "propose":
        target_words: Dict[str, int] = {}
        for t in ctx["target_texts"]:
            for w in _tokens(t):
                target_words[w] = target_words.get(w, 0) + 1
        neighbour_words = set()
        for t in ctx["neighbour_texts"]:
            neighbour_words |= _tokens(t)
        distinctive = [w for w, _ in sorted(target_words.items(), key=lambda kv: -kv[1])
                       if w not in neighbour_words]
        if not distinctive:
            distinctive = [w for w, _ in sorted(target_words.items(), key=lambda kv: -kv[1])]
        cands = []
        for i in range(ctx["n_out"]):
            words = distinctive[i:i + 2] or distinctive[:2] or ["misc", "items"]
            label = " ".join(w.capitalize() for w in words) or "Misc"
            cands.append({"label": label, "description": f"Items mentioning {' / '.join(words)}.",
                          "rationale": "mock: distinctive word overlap vs neighbours"})
        return {"candidates": cands}
    if kind == "verify":
        label_words = _tokens(ctx["label"]) | _tokens(ctx["description"])
        fits = []
        for t in ctx["items"]:
            ov = len(_tokens(t) & label_words)
            fits.append(ov > 0)
        return {"fits": fits}
    if kind == "refine":
        return {"label": ctx["label"], "description": ctx["description"],
                "rationale": "mock: no-op refine"}
    if kind == "stability":
        return _mock_response("propose", {**ctx, "n_out": 1})["candidates"][0]
    if kind == "subtheme":
        return {"names": [f"sub-theme {i + 1}" for i in range(len(ctx["groups"]))]}
    if kind == "decompose":
        # invariant = tokens shared by ALL target members and not in neighbours;
        # varying = a couple of axes from the most distinctive non-shared tokens.
        per_item = [_tokens(t) for t in ctx["target_texts"]]
        shared = set.intersection(*per_item) if per_item else set()
        neighbour_words = set()
        for t in ctx["neighbour_texts"]:
            neighbour_words |= _tokens(t)
        distinctive_shared = sorted(shared - neighbour_words)
        freq: Dict[str, int] = {}
        for toks in per_item:
            for w in toks:
                if w not in shared:
                    freq[w] = freq.get(w, 0) + 1
        varying_vals = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:6]
        invariant = [{"axis": "shared term", "value": v} for v in distinctive_shared[:2]]
        varying = [{"axis": "wording", "values": varying_vals, "open_ended": True}] if varying_vals else []
        return {"invariant_axes": invariant, "varying_axes": varying}
    if kind == "breadth_verify":
        inv_words = set()
        for a in ctx.get("invariant_axes", []):
            inv_words |= _tokens(str(a.get("value", "")))
        return {"fits": [len(_tokens(t) & inv_words) > 0 if inv_words else True for t in ctx["items"]]}
    if kind == "decompose_gap":
        return {"invariant_axes": [], "varying_axes": []}   # mock found everything in pass 1
    if kind == "invariant_summary":
        return {"summary": _describe_invariant(ctx.get("axes", []))}
    if kind == "varying_summary":
        return {"summary": _describe_varying(ctx.get("axes", []))}
    if kind == "redifferentiate":
        return {"a": {"label": ctx["a_label"] + " (A)", "description": ctx["a_desc"]},
                "b": {"label": ctx["b_label"] + " (B)", "description": ctx["b_desc"]}}
    return {}


# ===========================================================================
# evidence construction
# ===========================================================================
@dataclass
class _Ctx:
    emb_n: np.ndarray
    cent: np.ndarray
    nb_order: np.ndarray
    text_arr: np.ndarray
    idxs_by_code: List[np.ndarray]
    cfg: LabelConfig
    client: _LLMClient
    K: int
    report: _Reporter


def _micro_mode_medoids(emb_n: np.ndarray, idxs: np.ndarray, k: int, seed: int) -> Tuple[List[int], float]:
    """k representative medoids spanning the item's sub-modes, plus a spread ratio
    (mean inter-medoid distance / mean intra-mode distance) used for sub-theme detection."""
    n = len(idxs)
    k = max(1, min(k, n))
    if n <= k or n < 4:
        return list(map(int, idxs)), 1.0
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=1).fit(emb_n[idxs])
    medoids, intra = [], []
    for m in range(k):
        mem = idxs[km.labels_ == m]
        if len(mem) == 0:
            continue
        c = km.cluster_centers_[m]
        d = np.linalg.norm(emb_n[mem] - c, axis=1)
        medoids.append(int(mem[int(np.argmin(d))]))
        intra.append(float(d.mean()))
    if len(medoids) < 2:
        return medoids, 1.0
    mc = _normalize(km.cluster_centers_[: len(medoids)])
    inter = 1 - (mc @ mc.T)
    iu = np.triu_indices(len(medoids), k=1)
    inter_mean = float(inter[iu].mean()) if len(iu[0]) else 0.0
    intra_mean = float(np.mean(intra)) or 1e-6
    return medoids, inter_mean / intra_mean


def _build_evidence(code: int, train_idxs: np.ndarray, ctx: _Ctx, rng: np.random.Generator) -> dict:
    cfg = ctx.cfg
    sims = ctx.emb_n[train_idxs] @ ctx.cent[code]
    order = np.argsort(-sims)
    core = [int(train_idxs[i]) for i in order[: min(cfg.n_core, len(train_idxs))]]

    diverse_pool = np.setdiff1d(train_idxs, np.asarray(core))
    diverse, spread_ratio = (_micro_mode_medoids(ctx.emb_n, diverse_pool, cfg.micro_k, cfg.seed)
                              if len(diverse_pool) else ([], 1.0))
    diverse = diverse[: cfg.n_diverse]

    neighbour_codes = [int(x) for x in ctx.nb_order[code][: min(cfg.n_contrast_clusters, ctx.K - 1)]]
    boundary: List[int] = []
    if neighbour_codes:
        nb_cent = _normalize(ctx.cent[neighbour_codes].mean(axis=0, keepdims=True))[0]
        bsims = ctx.emb_n[train_idxs] @ nb_cent
        bord = np.argsort(-bsims)[: cfg.n_boundary]
        boundary = [int(train_idxs[i]) for i in bord]

    return dict(core=core, diverse=diverse, boundary=boundary,
                neighbour_codes=neighbour_codes, spread_ratio=spread_ratio)


def _neighbour_exemplars(ctx: _Ctx, neighbour_codes: Sequence[int], n_each: int,
                         rng: np.random.Generator) -> Tuple[List[str], List[int]]:
    texts, ids = [], []
    for nc in neighbour_codes:
        nidxs = ctx.idxs_by_code[nc]
        if len(nidxs) == 0:
            continue
        nsims = ctx.emb_n[nidxs] @ ctx.cent[nc]
        top = nidxs[np.argsort(-nsims)[: n_each]]
        for i in top:
            texts.append(_clip(ctx.text_arr[i], ctx.cfg.item_chars))
            ids.append(int(i))
    return texts, ids


def _sample_negatives(ctx: _Ctx, neighbour_codes: Sequence[int], n: int,
                      rng: np.random.Generator,
                      exclude: Optional[set] = None) -> Tuple[List[int], List[str]]:
    # negatives must be HELD OUT: drop any sibling items already shown to the
    # model as contrast exemplars during proposal, otherwise the precision
    # (sibling-rejection) score is measured on items the candidate already saw.
    exclude = exclude or set()
    pool: List[int] = []
    for nc in neighbour_codes:
        pool.extend(int(i) for i in ctx.idxs_by_code[nc] if int(i) not in exclude)
    if not pool:
        return [], []
    take = min(n, len(pool))
    chosen = list(rng.choice(np.asarray(pool), size=take, replace=False))
    return chosen, [_clip(ctx.text_arr[i], ctx.cfg.item_chars) for i in chosen]


# ===========================================================================
# prompts
# ===========================================================================
def _head(cfg: LabelConfig) -> str:
    h = ""
    if cfg.domain_hint:
        h += f"# domain: {cfg.domain_hint}\n"
    if cfg.same_when:
        h += f"# rule: two items are the same kind when {cfg.same_when}\n"
    return h


def _axes_guidance(invariant_axes: List[dict], varying_axes: List[dict]) -> str:
    """Render the axis decomposition as guidance for the proposer (additive)."""
    if not invariant_axes and not varying_axes:
        return ""
    inv = ", ".join(f"{a.get('axis')}={a.get('value')}" for a in invariant_axes) or "(none found)"
    var = ", ".join(str(a.get("axis")) for a in varying_axes) or "(none found)"
    return ("SHARED IDENTITY (true of EVERY target, what separates it from neighbours — name the "
            f"label from THESE): {inv}\n"
            f"INCIDENTAL VARIATION (differs across targets — do NOT name the label after these): {var}\n\n")


def _propose_prompt(cfg: LabelConfig, target_texts: List[str], neighbour_texts: List[str], n_out: int,
                    invariant_axes: Optional[List[dict]] = None,
                    varying_axes: Optional[List[dict]] = None) -> str:
    return (_head(cfg) +
            _axes_guidance(invariant_axes or [], varying_axes or []) +
            f"TARGET ITEMS (core + sub-modes of one cluster):\n{_numbered(target_texts)}\n\n"
            f"NEIGHBOUR ITEMS (sampled from the nearest OTHER clusters - what the target is NOT):\n"
            f"{_numbered(neighbour_texts)}\n\n"
            f"Propose {n_out} DISTINCT candidate cards for the TARGET cluster. Each card's label+"
            "description must be TRUE of every target item and FALSE of the neighbour items - name "
            "the attribute that separates them, not a topic word they share. Candidates must differ "
            "from each other in wording or angle, not be paraphrases.\n"
            'Return STRICT JSON: {"candidates":[{"label":"<=8 words","description":"<=25 words",'
            '"rationale":"<=20 words, what separates target from neighbours"}]}')


def _decompose_prompt(cfg: LabelConfig, target_texts: List[str], neighbour_texts: List[str]) -> str:
    return (_head(cfg) +
            f"TARGET MEMBERS (a diverse sample of ONE cluster):\n{_numbered(target_texts)}\n\n"
            f"NEIGHBOUR MEMBERS (from the nearest OTHER clusters):\n{_numbered(neighbour_texts)}\n\n"
            "Decompose the TARGET cluster into the dimensions on which its members AGREE vs DIFFER:\n"
            "(a) invariant_axes: attributes shared by EVERY target member AND that separate them "
            "from the neighbours (the cluster's shared identity). Each as {axis, value}.\n"
            "(b) varying_axes: attributes that DIFFER across target members. Each as {axis, values "
            "(the distinct values you observe), open_ended (true if the list is illustrative, not "
            "exhaustive)}.\n"
            'Return STRICT JSON: {"invariant_axes":[{"axis":"...","value":"..."}],'
            '"varying_axes":[{"axis":"...","values":["..."],"open_ended":false}]}')


def _breadth_verify_prompt(cfg: LabelConfig, invariant_axes: List[dict], items: List[str]) -> str:
    axes = "; ".join(f"{a.get('axis')}: {a.get('value')}" for a in invariant_axes) or "(none)"
    return (_head(cfg) +
            f"A cluster's SHARED IDENTITY is defined by these invariant attributes:\n{axes}\n\n"
            f"For each numbered item below, does it satisfy ALL of those attributes? Judge each "
            f"independently and strictly.\nITEMS:\n{_numbered(items)}\n"
            'Return STRICT JSON: {"fits":[true,false,...]} (one entry per item, in order)')


def _gap_prompt(cfg: LabelConfig, invariant_axes: List[dict], varying_axes: List[dict],
                items: List[str]) -> str:
    inv = ", ".join(f"{a.get('axis')}={a.get('value')}" for a in invariant_axes) or "(none yet)"
    var = "; ".join(f"{a.get('axis')} ({', '.join(str(v) for v in (a.get('values') or []))})"
                    for a in varying_axes) or "(none yet)"
    return (_head(cfg) +
            "We are decomposing ONE cluster into invariant axes (attributes shared by ALL members) "
            "and varying axes (dimensions members differ on). Axes found so far:\n"
            f"INVARIANT: {inv}\nVARYING: {var}\n\n"
            f"Here are ADDITIONAL members, including edge cases / outliers:\n{_numbered(items)}\n\n"
            "List ONLY axes MISSING from the lists above — new attributes shared by these members "
            "too, or new dimensions members differ on. Return empty lists if nothing is missing.\n"
            'Return STRICT JSON: {"invariant_axes":[{"axis":"...","value":"..."}],'
            '"varying_axes":[{"axis":"...","values":["..."],"open_ended":false}]}')


def _invariant_summary_prompt(cfg: LabelConfig, invariant_axes: List[dict], items: List[str]) -> str:
    axes = ", ".join(f"{a.get('axis')}={a.get('value')}" for a in invariant_axes) or "(none)"
    return (_head(cfg) +
            f"These are members of ONE cluster:\n{_numbered(items)}\n\n"
            f"They all share these attributes: {axes}.\n"
            "In 1-2 natural sentences, describe what EVERY member of this cluster shares — its "
            "common identity. Be specific and concrete; do not list members or hedge.\n"
            'Return STRICT JSON: {"summary":"..."}')


def _varying_summary_prompt(cfg: LabelConfig, varying_axes: List[dict], items: List[str]) -> str:
    axes = "; ".join(f"{a.get('axis')} ({', '.join(str(v) for v in (a.get('values') or []))})"
                     for a in varying_axes) or "(none)"
    return (_head(cfg) +
            f"These are members of ONE cluster:\n{_numbered(items)}\n\n"
            f"They differ along these dimensions: {axes}.\n"
            "In 1-2 natural sentences, describe the range of variation across members — how they "
            "differ from one another. Be specific and concrete; do not list members or hedge.\n"
            'Return STRICT JSON: {"summary":"..."}')


def _verify_prompt(cfg: LabelConfig, label: str, description: str, items: List[str]) -> str:
    return (_head(cfg) +
            f"CARD - LABEL: {label}\nDESCRIPTION: {description}\n\n"
            f"For each numbered item below, does it FIT the card (true member)? Items are a mix; "
            f"judge each independently and strictly - do not assume most will fit.\n"
            f"ITEMS:\n{_numbered(items)}\n"
            'Return STRICT JSON: {"fits":[true,false,...]} (one entry per item, in order)')


def _refine_prompt(cfg: LabelConfig, label: str, description: str,
                   false_negatives: List[str], false_positives: List[str]) -> str:
    fn = _numbered(false_negatives) if false_negatives else "(none)"
    fp = _numbered(false_positives) if false_positives else "(none)"
    return (_head(cfg) +
            f"CURRENT CARD - LABEL: {label}\nDESCRIPTION: {description}\n\n"
            f"This card WRONGLY REJECTED these true members (should fit but didn't, broaden if needed):\n{fn}\n\n"
            f"This card WRONGLY ACCEPTED these non-members (should NOT fit but did, narrow/sharpen if needed):\n{fp}\n\n"
            "Revise the label and description to fix both kinds of error.\n"
            'Return STRICT JSON: {"label":"<=8 words","description":"<=25 words","rationale":"<=20 words"}')


def _subtheme_prompt(cfg: LabelConfig, groups: List[List[str]]) -> str:
    body = "\n\n".join(f"GROUP {i + 1}:\n{_numbered(g)}" for i, g in enumerate(groups))
    return (_head(cfg) +
            f"These look like sub-themes within one cluster. Name each group specifically (<=6 words):\n{body}\n"
            'Return STRICT JSON: {"names":["...", "..."]}')


def _redifferentiate_prompt(cfg: LabelConfig, a_label, a_desc, a_items, b_label, b_desc, b_items) -> str:
    return (_head(cfg) +
            "Two clusters currently have near-identical labels but are different clusters. Revise BOTH "
            "labels/descriptions so they are clearly distinct from each other, each still accurate to its items.\n"
            f"CLUSTER A - LABEL: {a_label}\nDESCRIPTION: {a_desc}\nITEMS:\n{_numbered(a_items)}\n\n"
            f"CLUSTER B - LABEL: {b_label}\nDESCRIPTION: {b_desc}\nITEMS:\n{_numbered(b_items)}\n"
            'Return STRICT JSON: {"a":{"label":"...","description":"..."},'
            '"b":{"label":"...","description":"..."}}')


# ===========================================================================
# stage 2/3: verification + refinement
# ===========================================================================
_NO_METRICS = {"recall": None, "precision": None, "specificity": None, "discrimination": None}


def _disc(metrics: dict) -> float:
    return metrics.get("discrimination") or 0.0


def _fmt_score(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def _grade_candidate(ctx: _Ctx, cand: dict, pos_texts: List[str], neg_texts: List[str],
                     rng: np.random.Generator) -> Tuple[dict, List[bool]]:
    """Grade a candidate as a classifier over held-out positives + sibling negatives.

    Returns a metrics dict and the per-item verdicts (in original pos-then-neg
    order). The metrics are reported under their correct statistical names:
      recall        = held-out members accepted          TP / (TP+FN)
      precision     = of items it accepts, share correct TP / (TP+FP)
      specificity   = sibling items correctly rejected   TN / (TN+FP)
      discrimination= balanced accuracy of recall+specificity
    """
    cfg = ctx.cfg
    none_metrics = {"recall": None, "precision": None, "specificity": None, "discrimination": None}
    items = pos_texts + neg_texts
    if not items:
        return none_metrics, []
    # Shuffle members and siblings together before grading: presenting all
    # positives first then all negatives is a positional tell that lets the
    # judge infer the boundary instead of judging each item on its merits.
    perm = rng.permutation(len(items))
    shuffled = [items[i] for i in perm]
    res = ctx.client.complete(
        _verify_prompt(cfg, cand["label"], cand.get("description", ""), shuffled),
        "verify", {"label": cand["label"], "description": cand.get("description", ""), "items": shuffled})
    shuffled_fits = _fits_of(res, len(items))
    fits = [False] * len(items)                       # un-shuffle back to pos-then-neg order
    for shuf_pos, orig_idx in enumerate(perm):
        fits[orig_idx] = shuffled_fits[shuf_pos]
    n_pos, n_neg = len(pos_texts), len(neg_texts)
    tp = sum(1 for f in fits[:n_pos] if f)
    fp = sum(1 for f in fits[n_pos:] if f)
    recall = tp / n_pos if n_pos else None
    specificity = (n_neg - fp) / n_neg if n_neg else None
    precision = tp / (tp + fp) if (tp + fp) else None      # undefined when nothing is accepted
    parts = [m for m in (recall, specificity) if m is not None]
    discrimination = sum(parts) / len(parts) if parts else None
    return {"recall": recall, "precision": precision,
            "specificity": specificity, "discrimination": discrimination}, fits


def _refine_loop(ctx: _Ctx, cand: dict, pos_texts: List[str], pos_ids: List[int],
                 neg_texts: List[str], neg_ids: List[int], rng: np.random.Generator) -> Tuple[dict, dict]:
    cfg = ctx.cfg
    best_cand, best_metrics, best_fits = cand, *_grade_candidate(ctx, cand, pos_texts, neg_texts, rng)
    n_pos = len(pos_texts)
    for _ in range(cfg.refine_max_iters):
        if _disc(best_metrics) >= cfg.accept_discrimination:
            break
        # reuse the verdicts from the grading we already paid for instead of
        # re-running verify on the same items (halves the calls per iteration
        # and keeps false-pos/neg consistent with the score that drove us here).
        false_neg = [pos_texts[i] for i, f in enumerate(best_fits[:n_pos]) if not f]
        false_pos = [neg_texts[i] for i, f in enumerate(best_fits[n_pos:]) if f]
        if not false_neg and not false_pos:
            break
        rev = ctx.client.complete(
            _refine_prompt(cfg, best_cand["label"], best_cand.get("description", ""), false_neg, false_pos),
            "refine", {"label": best_cand["label"], "description": best_cand.get("description", "")})
        if not isinstance(rev, dict) or not rev.get("label"):
            break
        new_cand = {"label": rev["label"], "description": rev.get("description", ""),
                    "rationale": rev.get("rationale", best_cand.get("rationale", ""))}
        m2, f2 = _grade_candidate(ctx, new_cand, pos_texts, neg_texts, rng)
        if _disc(m2) >= _disc(best_metrics):
            best_cand, best_metrics, best_fits = new_cand, m2, f2
        else:
            break
    return best_cand, best_metrics


# ===========================================================================
# stage 4: stability
# ===========================================================================
def _stability_score(ctx: _Ctx, code: int, train_idxs: np.ndarray, base_label: str,
                     rng: np.random.Generator) -> Optional[float]:
    # None means "not assessed" (resampling disabled or too few items) — distinct
    # from a measured 1.0. Reported as n/a and treated as neutral by the confidence
    # band, so a small cluster isn't silently credited with perfect stability.
    cfg = ctx.cfg
    if cfg.stability_resamples <= 0 or len(train_idxs) < cfg.n_core + 2:
        return None
    sims_total = []
    for _ in range(cfg.stability_resamples):
        resample = rng.choice(train_idxs, size=min(len(train_idxs), max(cfg.n_core, len(train_idxs) // 2)),
                              replace=False)
        ev = _build_evidence(code, resample, ctx, rng)
        target_texts = [_clip(ctx.text_arr[i], cfg.item_chars) for i in (ev["core"] + ev["diverse"])]
        neighbour_texts, _ = _neighbour_exemplars(ctx, ev["neighbour_codes"], cfg.n_contrast_items, rng)
        res = ctx.client.complete(
            _propose_prompt(cfg, target_texts, neighbour_texts, 1),
            "stability", {"target_texts": target_texts, "neighbour_texts": neighbour_texts, "n_out": 1})
        cands = _candidates_of(res)
        if cands:
            sims_total.append(_jaccard(base_label, cands[0].get("label", "")))
    return float(np.mean(sims_total)) if sims_total else None


# ===========================================================================
# stage 5: sub-themes (informational)
# ===========================================================================
def _maybe_subthemes(ctx: _Ctx, idxs: np.ndarray, spread_ratio: float, rng: np.random.Generator) -> Optional[List[dict]]:
    cfg = ctx.cfg
    if spread_ratio < cfg.subtheme_spread_ratio or len(idxs) < cfg.subtheme_min_size:
        return None
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=2, random_state=cfg.seed, n_init=1).fit(ctx.emb_n[idxs])
    groups, group_ids = [], []
    for m in range(2):
        mem = idxs[km.labels_ == m]
        if len(mem) == 0:
            continue
        c = km.cluster_centers_[m]
        order = np.argsort(np.linalg.norm(ctx.emb_n[mem] - c, axis=1))[:6]
        group_ids.append([int(x) for x in mem[order]])
        groups.append([_clip(ctx.text_arr[i], cfg.item_chars) for i in mem[order]])
    if len(groups) < 2:
        return None
    res = ctx.client.complete(_subtheme_prompt(cfg, groups), "subtheme", {"groups": groups})
    names = (res.get("names") if isinstance(res, dict) else None) or \
            [f"sub-theme {i + 1}" for i in range(len(groups))]
    return [{"name": names[i] if i < len(names) else f"sub-theme {i + 1}",
             "size": int((km.labels_ == i).sum()), "exemplar_ids": group_ids[i]}
            for i in range(len(groups))]


# ===========================================================================
# breadth: invariant vs varying axes (computed BEFORE the label, used to guide it)
# ===========================================================================
def _union_axes(a: List[dict], b: List[dict], *, merge_values: bool) -> List[dict]:
    """Union two axis lists, deduping by lowercased axis name. For varying axes,
    merge their value sets; for invariant axes, keep the first value seen."""
    by_name: Dict[str, dict] = {}
    order: List[str] = []
    for ax in list(a) + list(b):
        if not isinstance(ax, dict) or not ax.get("axis"):
            continue
        key = str(ax["axis"]).strip().lower()
        if key not in by_name:
            by_name[key] = dict(ax)
            order.append(key)
        elif merge_values:
            existing = [str(x) for x in (by_name[key].get("values") or [])]
            new = [str(x) for x in (ax.get("values") or [])]
            by_name[key]["values"] = list(dict.fromkeys(existing + new))
            by_name[key]["open_ended"] = bool(by_name[key].get("open_ended")) or bool(ax.get("open_ended"))
    return [by_name[k] for k in order]


def _describe_invariant(axes: List[dict]) -> str:
    """Descriptive collation of the invariant axes (what every member shares)."""
    parts = [f"{a.get('axis')} ({a.get('value')})" for a in axes if a.get("axis") and a.get("value")]
    return ("All members share " + ", ".join(parts) + ".") if parts else ""


def _describe_varying(axes: List[dict]) -> str:
    """Descriptive collation of the varying axes (how members differ)."""
    parts = []
    for a in axes:
        if not a.get("axis"):
            continue
        vals = ", ".join(str(v) for v in (a.get("values") or []))
        suffix = ", …" if a.get("open_ended") else ""
        parts.append(f"{a['axis']} ({vals}{suffix})" if vals else str(a["axis"]))
    return ("Members vary by " + "; ".join(parts) + ".") if parts else ""


def _extreme_sample(ctx: _Ctx, member_idxs: np.ndarray, code: int, exclude: set, k: int,
                    rng: np.random.Generator) -> List[int]:
    """A fresh sample weighted toward the cluster's EDGES — members farthest from the
    centroid (where missed axes hide) plus a random draw — excluding already-shown ids."""
    pool = np.asarray([int(i) for i in member_idxs if int(i) not in exclude])
    if len(pool) == 0:
        return []
    sims = ctx.emb_n[pool] @ ctx.cent[code]
    order = np.argsort(sims)                      # ascending -> farthest from centroid first
    n_out = min(len(pool), max(1, k // 2))
    out = [int(i) for i in pool[order[:n_out]]]
    remaining = [int(i) for i in pool if int(i) not in set(out)]
    n_rand = min(len(remaining), max(0, k - len(out)))
    rand = [int(i) for i in rng.choice(np.asarray(remaining), size=n_rand, replace=False)] if n_rand else []
    return out + rand


def _decompose_axes(ctx: _Ctx, ev: dict, neighbour_texts: List[str], rng: np.random.Generator,
                    *, member_idxs: np.ndarray, code: int) -> dict:
    """Decompose a cluster into invariant axes (shared identity, distinctive vs
    neighbours) + varying axes (the spread). Computed from a DIVERSE sample so all
    sub-modes are represented. `invariant_summary` / `varying_summary` are written as
    natural prose via two focused LLM asks (or a deterministic collation of the axes
    when breadth_prose is off); `coherence` is filled later by verifying the
    invariants on held-out members."""
    cfg = ctx.cfg
    # diverse sample: prioritise sub-mode medoids, then core, then boundary; dedupe.
    pool, seen = [], set()
    for i in list(ev.get("diverse", [])) + list(ev.get("core", [])) + list(ev.get("boundary", [])):
        if i not in seen:
            seen.add(i)
            pool.append(int(i))

    sample_texts = [_clip(ctx.text_arr[i], cfg.item_chars) for i in pool[: cfg.breadth_exemplars]]

    def _extract(texts: List[str]):
        res = ctx.client.complete(_decompose_prompt(cfg, texts, neighbour_texts),
                                  "decompose", {"target_texts": texts, "neighbour_texts": neighbour_texts})
        if not isinstance(res, dict):
            return [], []
        inv = [a for a in (res.get("invariant_axes") or []) if isinstance(a, dict) and a.get("axis")]
        var = [a for a in (res.get("varying_axes") or []) if isinstance(a, dict) and a.get("axis")]
        return inv, var

    invariant, varying = _extract(sample_texts)
    # robustness: union independent extractions on resampled evidence to catch axes
    # a single pass missed.
    for _ in range(max(0, cfg.breadth_resamples - 1)):
        if len(pool) <= cfg.breadth_exemplars:
            break
        ids = [int(i) for i in rng.choice(np.asarray(pool), size=cfg.breadth_exemplars, replace=False)]
        inv2, var2 = _extract([_clip(ctx.text_arr[i], cfg.item_chars) for i in ids])
        invariant = _union_axes(invariant, inv2, merge_values=False)
        varying = _union_axes(varying, var2, merge_values=True)

    # GAP PASS: show the axes found so far + a FRESH, edge-weighted sample and ask ONLY
    # for what's MISSING, then union. Higher marginal recall than blind resampling; stop
    # early once a pass adds nothing (saturation).
    shown = set(pool)
    for _ in range(max(0, cfg.breadth_gap_passes)):
        gap_ids = _extreme_sample(ctx, member_idxs, code, shown, cfg.breadth_exemplars, rng)
        if not gap_ids:
            break
        shown.update(gap_ids)
        res = ctx.client.complete(
            _gap_prompt(cfg, invariant, varying, [_clip(ctx.text_arr[i], cfg.item_chars) for i in gap_ids]),
            "decompose_gap", {"target_texts": [_clip(ctx.text_arr[i], cfg.item_chars) for i in gap_ids],
                              "invariant_axes": invariant, "varying_axes": varying})
        inv_new = [a for a in (res.get("invariant_axes") or []) if isinstance(a, dict) and a.get("axis")] \
            if isinstance(res, dict) else []
        var_new = [a for a in (res.get("varying_axes") or []) if isinstance(a, dict) and a.get("axis")] \
            if isinstance(res, dict) else []
        before = len(invariant) + len(varying)
        invariant = _union_axes(invariant, inv_new, merge_values=False)
        varying = _union_axes(varying, var_new, merge_values=True)
        if len(invariant) + len(varying) == before:
            break                                # saturation: nothing new found

    varying = varying[: cfg.breadth_max_axes]
    example_ids = [int(i) for i in list(ev.get("diverse", []))[:4]]
    for v in varying:
        v["values"] = [str(x) for x in (v.get("values") or [])]
        v["open_ended"] = bool(v.get("open_ended", False))
        v.setdefault("example_ids", example_ids)

    # Two FOCUSED asks: one natural-prose summary of what's shared, one of how members
    # vary. Falls back to the deterministic collation when prose is off or the model
    # returns nothing (which is also what the offline mock yields, keeping it stable).
    def _prose(axes: List[dict], prompt: str, kind: str, fallback: str) -> str:
        if axes and cfg.breadth_prose:
            res = ctx.client.complete(prompt, kind, {"axes": axes})
            text = res.get("summary", "") if isinstance(res, dict) else ""
            if text:
                return _clip(text, cfg.breadth_summary_chars)
        return _clip(fallback, cfg.breadth_summary_chars)

    inv_summary = _prose(invariant, _invariant_summary_prompt(cfg, invariant, sample_texts),
                         "invariant_summary", _describe_invariant(invariant))
    var_summary = _prose(varying, _varying_summary_prompt(cfg, varying, sample_texts),
                         "varying_summary", _describe_varying(varying))
    return {"invariant_summary": inv_summary, "varying_summary": var_summary,
            "invariant_axes": invariant, "varying_axes": varying,
            "coherence": None, "n_invariant": len(invariant), "n_varying": len(varying)}


def _breadth_coherence(ctx: _Ctx, invariant_axes: List[dict], holdout_texts: List[str]) -> Optional[float]:
    """Fraction of held-out members consistent with ALL invariant axes."""
    cfg = ctx.cfg
    if not (cfg.breadth_verify and invariant_axes and holdout_texts):
        return None
    res = ctx.client.complete(_breadth_verify_prompt(cfg, invariant_axes, holdout_texts),
                              "breadth_verify", {"invariant_axes": invariant_axes, "items": holdout_texts})
    fits = _fits_of(res, len(holdout_texts))
    return sum(1 for f in fits if f) / len(fits) if fits else None


# ===========================================================================
# per-cluster pipeline
# ===========================================================================
def _label_one_cluster(code: int, cid: str, ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    idxs = ctx.idxs_by_code[code]
    n = len(idxs)
    rng = np.random.default_rng([cfg.seed, code])
    ctx.client.reset_call_counter()
    # Buffer this worker's stage messages and attach them to the returned card.
    # With concurrent workers, printing live interleaves clusters; the caller
    # flushes each cluster's lines as one atomic block when it finishes.
    log_lines: List[Tuple[int, str]] = []
    def say(msg: str, level: int = 2) -> None:
        log_lines.append((level, msg))

    say(f"  · start (size {n})")

    if n < cfg.min_cluster_size:
        ev = _build_evidence(code, idxs, ctx, rng)
        target_texts = [_clip(ctx.text_arr[i], cfg.item_chars) for i in (ev["core"] + ev["diverse"] + ev["boundary"])]
        neighbour_texts, _ = _neighbour_exemplars(ctx, ev["neighbour_codes"], cfg.n_contrast_items, rng)
        say(f"  · small cluster — labeling from all {n} members, no held-out check")
        breadth = _decompose_axes(ctx, ev, neighbour_texts, rng, member_idxs=idxs, code=code)
        res = ctx.client.complete(
            _propose_prompt(cfg, target_texts, neighbour_texts, 1,
                            breadth["invariant_axes"], breadth["varying_axes"]),
            "propose", {"target_texts": target_texts, "neighbour_texts": neighbour_texts, "n_out": 1})
        cands = _candidates_of(res)
        cand = cands[0] if cands else {"label": cid, "description": "", "rationale": ""}
        card = _finalize(ctx, cid, cand, n, metrics=dict(_NO_METRICS),
                         stability=None, evidence=ev, alternatives=[], subthemes=None, breadth=breadth,
                         note="cluster too small for held-out verification",
                         n_calls=ctx.client.calls_since_reset())
        card["_log"] = log_lines
        return card

    # held-out split: never shown to evidence/proposal/refine
    perm = rng.permutation(idxs)
    n_hold = min(max(cfg.min_holdout, int(round(n * cfg.holdout_frac))), n - 3)
    holdout, train = perm[:n_hold], perm[n_hold:]

    ev = _build_evidence(code, train, ctx, rng)
    target_texts = [_clip(ctx.text_arr[i], cfg.item_chars) for i in (ev["core"] + ev["diverse"] + ev["boundary"])]
    neighbour_texts, shown_neighbour_ids = _neighbour_exemplars(ctx, ev["neighbour_codes"], cfg.n_contrast_items, rng)
    say(f"  · evidence: {len(ev['core'])} core + {len(ev['diverse'])} diverse + "
        f"{len(ev['boundary'])} boundary; {len(holdout)} held out; "
        f"{len(ev['neighbour_codes'])} contrast clusters")

    # DECOMPOSE first: invariant axes (the shared identity the label should name) +
    # varying axes (the spread). This guides PROPOSE so the label names the essence,
    # not an incidental varying attribute.
    breadth = _decompose_axes(ctx, ev, neighbour_texts, rng, member_idxs=train, code=code)
    say(f"  · axes: {breadth['n_invariant']} shared / {breadth['n_varying']} varying")

    res = ctx.client.complete(
        _propose_prompt(cfg, target_texts, neighbour_texts, cfg.n_candidates,
                        breadth["invariant_axes"], breadth["varying_axes"]),
        "propose", {"target_texts": target_texts, "neighbour_texts": neighbour_texts,
                    "n_out": cfg.n_candidates})
    candidates = _candidates_of(res) or [{"label": cid, "description": "", "rationale": ""}]
    say(f"  · proposed {len(candidates)} candidate(s): "
        f"{', '.join(repr(c.get('label', '')) for c in candidates[:4])}")

    # Grade on a CAPPED sample of held-out members. The full holdout can be ~30%
    # of a huge cluster (thousands of items); sending all of them in one verify
    # prompt is what makes large clusters time out (and truncated 'fits' arrays
    # corrupt recall). The whole holdout is still excluded from evidence/proposal.
    pos_pool = holdout
    if len(pos_pool) > cfg.verify_positives:
        pos_pool = rng.choice(pos_pool, size=cfg.verify_positives, replace=False)
    pos_texts = [_clip(ctx.text_arr[i], cfg.item_chars) for i in pos_pool]
    neg_ids, neg_texts = _sample_negatives(ctx, ev["neighbour_codes"], cfg.verify_negatives, rng,
                                           exclude=set(shown_neighbour_ids))
    say(f"  · verifying on {len(pos_texts)} held-out members + {len(neg_texts)} sibling negatives "
        f"(of {len(holdout)} held out)")

    graded = []
    for cand in candidates:
        best_cand, metrics = _refine_loop(ctx, cand, pos_texts, list(pos_pool), neg_texts, neg_ids, rng)
        graded.append((best_cand, metrics))
        say(f"  · graded {best_cand.get('label', '')!r}: "
            f"disc={_fmt_score(metrics.get('discrimination'))} "
            f"rec={_fmt_score(metrics.get('recall'))} "
            f"spec={_fmt_score(metrics.get('specificity'))}")
    graded.sort(key=lambda g: -_disc(g[1]))
    best_cand, best_metrics = graded[0]
    alternatives = [{"label": g[0]["label"], "description": g[0].get("description", ""),
                     "discrimination": g[1].get("discrimination")} for g in graded[1:]]

    stability = _stability_score(ctx, code, train, best_cand["label"], rng)
    subthemes = _maybe_subthemes(ctx, idxs, ev["spread_ratio"], rng)
    if subthemes:
        say(f"  · sub-themes detected: {', '.join(repr(s['name']) for s in subthemes)}")

    # verify the invariant axes on held-out members -> coherence (low = the cluster
    # lacks a real shared identity).
    breadth["coherence"] = _breadth_coherence(ctx, breadth["invariant_axes"], pos_texts)
    say(f"  · breadth: {breadth['n_invariant']} invariant / {breadth['n_varying']} varying axes, "
        f"coherence {_fmt_score(breadth['coherence'])}")

    # When a cluster has no siblings (K == 1) there are no negatives, so precision
    # and specificity are unmeasured and discrimination is recall-only — flag it
    # rather than let a recall-only score masquerade as full discrimination.
    note = None if neg_texts else ("no sibling negatives: precision/specificity unmeasured, "
                                   "discrimination is recall-only")

    card = _finalize(ctx, cid, best_cand, n, metrics=best_metrics,
                     stability=stability, evidence=ev, alternatives=alternatives, subthemes=subthemes,
                     breadth=breadth, note=note, n_calls=ctx.client.calls_since_reset())
    card["_log"] = log_lines
    return card


def _confidence_band(cfg: LabelConfig, metrics: dict, stability: Optional[float]) -> str:
    disc = metrics.get("discrimination")
    if disc is None:
        return "unverified"
    recall, spec = metrics.get("recall"), metrics.get("specificity")
    stable = stability is None or stability >= cfg.stability_min_jaccard
    # "high" must clear every bar the config advertises, not discrimination alone:
    # a recall-0.45 / specificity-0.95 label scores discrimination 0.70 but misses
    # half its members, so accept_recall/accept_precision are enforced here too.
    recall_ok = recall is None or recall >= cfg.accept_recall
    # specificity must be MEASURED and pass: a label that never had a sibling to
    # reject (no negatives, e.g. K==1) has not earned "high", only recall-only.
    spec_ok = spec is not None and spec >= cfg.accept_precision
    if disc >= cfg.accept_discrimination and recall_ok and spec_ok and stable:
        return "high"
    if disc >= 0.6:
        return "medium"
    return "low"


def _empty_breadth() -> dict:
    return {"invariant_summary": "", "varying_summary": "", "invariant_axes": [], "varying_axes": [],
            "coherence": None, "n_invariant": 0, "n_varying": 0}


def _finalize(ctx: _Ctx, cid, cand, size, *, metrics, stability, evidence,
             alternatives, subthemes, note, n_calls, breadth=None) -> dict:
    cfg = ctx.cfg
    ev_block: Dict[str, Any] = {}
    for key in ("core", "diverse", "boundary"):
        ids = list(evidence.get(key, []))
        ev_block[key] = ids
        # carry the actual exemplar texts, not just row indices, so a human can
        # re-check the evidence behind a label without re-joining to the frame.
        ev_block[f"{key}_texts"] = [_clip(ctx.text_arr[i], cfg.item_chars) for i in ids]
    return {
        "cluster_id": cid,
        "label": cand.get("label", cid),
        "description": _clip(cand.get("description", ""), cfg.desc_chars),
        "rationale": cand.get("rationale", ""),
        "size": size,
        "scores": {
            "recall": metrics.get("recall"), "precision": metrics.get("precision"),
            "specificity": metrics.get("specificity"), "discrimination": metrics.get("discrimination"),
            "stability": stability, "confidence": _confidence_band(cfg, metrics, stability),
        },
        "alternatives": alternatives,
        "confusable_with": [],
        "subthemes": subthemes,
        "breadth": breadth or _empty_breadth(),
        "evidence": ev_block,
        "n_llm_calls": n_calls,
        "note": note,
    }


def _error_card(cid: str, size: int, err: str) -> dict:
    """Full-shaped placeholder so one failed cluster never aborts the batch or
    breaks the downstream coherence pass / dataframe / report."""
    return {
        "cluster_id": cid, "label": cid, "description": "", "rationale": "", "size": size,
        "scores": {"recall": None, "precision": None, "specificity": None,
                   "discrimination": None, "stability": None, "confidence": "error"},
        "alternatives": [], "confusable_with": [], "subthemes": None, "breadth": _empty_breadth(),
        "evidence": {"core": [], "core_texts": [], "diverse": [], "diverse_texts": [],
                     "boundary": [], "boundary_texts": []},
        "n_llm_calls": 0, "note": f"labeling failed: {err}",
    }


# ===========================================================================
# stage 6: global coherence pass
# ===========================================================================
def _global_coherence_pass(scorecards: Dict[str, dict], ctx: _Ctx, cid_of: Dict[int, str]) -> Dict[str, int]:
    cfg = ctx.cfg
    cids = list(cid_of.values())
    code_of = {c: i for i, c in cid_of.items()}
    sims = ctx.cent @ ctx.cent.T
    n = len(cids)
    tally = {"redifferentiated": 0, "confusable_pairs": 0}
    for i in range(n):
        for j in range(i + 1, n):  # each unordered pair is visited exactly once
            a_cid, b_cid = cids[i], cids[j]
            a, b = scorecards[a_cid], scorecards[b_cid]
            cent_sim = float(sims[i, j])
            label_sim = _jaccard(a["label"], b["label"])
            if label_sim >= cfg.dedup_label_jaccard and cent_sim < cfg.dedup_cent_sim:
                a_idxs = ctx.idxs_by_code[code_of[a_cid]]
                b_idxs = ctx.idxs_by_code[code_of[b_cid]]
                a_items = [_clip(ctx.text_arr[k], cfg.item_chars) for k in a_idxs[: cfg.n_core]]
                b_items = [_clip(ctx.text_arr[k], cfg.item_chars) for k in b_idxs[: cfg.n_core]]
                old_a, old_b = a["label"], b["label"]
                res = ctx.client.complete(
                    _redifferentiate_prompt(cfg, a["label"], a["description"], a_items,
                                            b["label"], b["description"], b_items),
                    "redifferentiate", {"a_label": a["label"], "a_desc": a["description"],
                                        "b_label": b["label"], "b_desc": b["description"]})
                if not isinstance(res, dict):
                    res = {}
                if res.get("a", {}).get("label"):
                    a["label"], a["description"] = res["a"]["label"], res["a"].get("description", a["description"])
                if res.get("b", {}).get("label"):
                    b["label"], b["description"] = res["b"]["label"], res["b"].get("description", b["description"])
                tally["redifferentiated"] += 1
                ctx.report(f"  ↺ re-differentiated near-identical labels: [{a_cid}] {old_a!r}/[{b_cid}] {old_b!r} "
                           f"→ {a['label']!r} / {b['label']!r}")
            elif cent_sim >= cfg.dedup_cent_sim and label_sim < cfg.dedup_label_jaccard:
                a.setdefault("confusable_with", []).append({"cluster_id": b_cid, "label": b["label"],
                                                            "cent_similarity": round(cent_sim, 3)})
                b.setdefault("confusable_with", []).append({"cluster_id": a_cid, "label": a["label"],
                                                            "cent_similarity": round(cent_sim, 3)})
                tally["confusable_pairs"] += 1
                ctx.report(f"  ⚠ confusable content: [{a_cid}] {a['label']!r} ~ [{b_cid}] {b['label']!r} "
                           f"(centroid sim {cent_sim:.2f})")
    return tally


# ===========================================================================
# public entry point
# ===========================================================================
def label_clusters(df: pd.DataFrame, embeddings: Optional[np.ndarray] = None,
                   text_col: str = "text", cluster_col: str = "cluster_id",
                   embedding_col: Optional[str] = None, cfg: Optional[LabelConfig] = None,
                   llm_fn: Optional[Callable] = None, progress: bool = True,
                   verbose: int = 1) -> Dict[str, dict]:
    """Generate contrastive, held-out-verified labels for each cluster in `df`.

    Args:
      df: rows with at least `text_col` and `cluster_col`.
      embeddings: (n_rows, d) array aligned to df, OR pass `embedding_col` instead.
      llm_fn: (messages, json_mode=True) -> str. If omitted, uses use_llm()'s registered
              gateway, or falls back to an offline mock (for testing/demos only).
      progress: show a tqdm progress bar (if tqdm is installed).
      verbose: how much to print to stderr while running, independent of logging config:
               0 = silent, 1 = banner + one line per finished cluster + final summary,
               2 = also per-stage detail (evidence / propose / grade / sub-themes).
               All messages are also emitted to the 'cluster_labeler' logger.
    Returns: {cluster_id: scorecard} (see module docstring for the fields).
    """
    cfg = cfg or LabelConfig()
    resolved_fn = llm_fn or _GATEWAY[0]
    if resolved_fn is None and not cfg.allow_mock:
        raise ValueError(
            "no LLM gateway registered. Call use_llm()/use_genai(fn), pass llm_fn=, or set "
            "LabelConfig(allow_mock=True) to use the offline mock labeler (testing/demos only).")
    df = df.reset_index(drop=True)
    for col in (text_col, cluster_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in DataFrame (have {list(df.columns)})")
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    text_arr = df[text_col].astype(str).to_numpy()
    if embeddings is None:
        if not embedding_col:
            raise ValueError("pass `embeddings` or `embedding_col`")
        if embedding_col not in df.columns:
            raise KeyError(f"embedding_col {embedding_col!r} not found in DataFrame")
        try:
            embeddings = np.vstack(df[embedding_col].to_list())
        except Exception as e:  # ragged / non-array cells give a cryptic numpy error otherwise
            raise ValueError(f"could not stack embedding_col {embedding_col!r} into a matrix "
                             f"(rows must be equal-length vectors): {e}") from e
    try:
        embeddings = np.asarray(embeddings, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"embeddings could not be read as a float array: {e}") from e
    if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
        raise ValueError(f"embeddings must be 2-D aligned to df: got shape {embeddings.shape} "
                         f"for {len(df)} rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or inf")
    emb_n = _normalize(embeddings)
    client = _LLMClient(cfg, resolved_fn)

    cats = pd.Categorical(df[cluster_col].astype(str))
    codes = cats.codes
    cid_of = {i: str(c) for i, c in enumerate(cats.categories)}
    K = len(cats.categories)

    cent = np.zeros((K, emb_n.shape[1]), dtype=np.float32)
    for c in range(K):
        cent[c] = emb_n[codes == c].mean(axis=0)
    cent = _normalize(cent)
    sims = cent @ cent.T
    np.fill_diagonal(sims, -2)
    nb_order = np.argsort(-sims, axis=1)
    idxs_by_code = [np.where(codes == c)[0] for c in range(K)]

    show_bars = bool(progress and tqdm)
    # ONE bar: clusters finished (determinate), with the running LLM-call count
    # and rate in its postfix. A second stacked bar spams a new line per update
    # in notebooks, so the call count rides along in this bar's suffix instead.
    bar = tqdm(total=K, desc="labeling", unit="cluster") if show_bars else None
    bar_lock = threading.Lock()
    client.progress_bar = bar
    client.progress_lock = bar_lock
    client._t0 = time.time()
    report = _Reporter(verbose, use_bar=bar is not None)
    ctx = _Ctx(emb_n=emb_n, cent=cent, nb_order=nb_order, text_arr=text_arr,
              idxs_by_code=idxs_by_code, cfg=cfg, client=client, K=K, report=report)

    mode = "MOCK (offline)" if client.mock else f"model={cfg.model}"
    sizes = [int(len(idxs_by_code[c])) for c in range(K)]
    report(f"cluster_labeler: labeling {K} clusters over {len(df):,} items "
           f"({mode}, workers={cfg.workers})")
    report(f"  cluster sizes: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}; "
           f"acceptance bars: disc≥{cfg.accept_discrimination} rec≥{cfg.accept_recall} "
           f"spec≥{cfg.accept_precision}")
    size_of = {cid_of[c]: sizes[c] for c in range(K)}
    scorecards: Dict[str, dict] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(cfg.workers, K))) as ex:
        futs = {ex.submit(_label_one_cluster, c, cid_of[c], ctx): cid_of[c] for c in range(K)}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                sc = fut.result()
            except Exception as e:  # one bad cluster must not abort the batch
                log.exception("labeling cluster %s failed", cid)
                sc = _error_card(cid, size_of[cid], str(e))
            scorecards[cid] = sc
            done += 1
            if bar is not None:
                with bar_lock:
                    bar.update(1)
                client._tick_bar(force=True)   # refresh call count on every completion
            # flush this cluster's whole block at once: header line (level 1)
            # followed by its buffered stage detail (level 2), so concurrent
            # workers never interleave their lines.
            stage_lines = sc.pop("_log", [])
            block = [_cluster_line(sc, done, K)]
            block += [m for (lvl, m) in stage_lines if verbose >= lvl]
            report("\n".join(block))
    report("global coherence pass across all clusters …")
    tally = _global_coherence_pass(scorecards, ctx, cid_of)  # may make more LLM calls
    if bar is not None:
        client._tick_bar(force=True)           # final call count (incl. coherence pass)
        bar.close()
    report(_summary_lines(scorecards, tally, client, time.time() - t0))
    return {cid_of[c]: scorecards[cid_of[c]] for c in range(K)}


def _cluster_line(sc: dict, done: int, total: int) -> str:
    """One-line live summary of a finished cluster."""
    s = sc["scores"]
    conf = s["confidence"]
    mark = {"high": "✓", "medium": "•", "low": "·", "unverified": "?", "error": "✗"}.get(conf, "·")
    head = f"{mark} [{done}/{total}] [{sc['cluster_id']}] {sc['label']!r}  size={sc['size']:,}  {conf.upper()}"
    if conf == "error":
        return f"{head}  ({sc['note']})"
    tail = (f"  disc={_fmt_score(s['discrimination'])} rec={_fmt_score(s['recall'])} "
            f"prec={_fmt_score(s.get('precision'))} spec={_fmt_score(s.get('specificity'))} "
            f"stab={_fmt_score(s['stability'])}  calls={sc['n_llm_calls']}")
    extra = ""
    b = sc.get("breadth") or {}
    if b.get("n_varying"):
        extra += f"  [axes {b.get('n_invariant', 0)} shared/{b['n_varying']} varying"
        extra += f", coherence {_fmt_score(b.get('coherence'))}]" if b.get("coherence") is not None else "]"
    if sc.get("note"):
        extra += f"  ({sc['note']})"
    return head + tail + extra


def _summary_lines(scorecards: Dict[str, dict], tally: Dict[str, int],
                   client: _LLMClient, elapsed: float) -> str:
    bands = ["high", "medium", "low", "unverified", "error"]
    counts = {b: 0 for b in bands}
    for sc in scorecards.values():
        counts[sc["scores"]["confidence"]] = counts.get(sc["scores"]["confidence"], 0) + 1
    n_sub = sum(1 for sc in scorecards.values() if sc.get("subthemes"))
    cohs = [sc.get("breadth", {}).get("coherence") for sc in scorecards.values()]
    cohs = [c for c in cohs if c is not None]
    n_low_coh = sum(1 for c in cohs if c < 0.7)
    lines = [
        f"cluster_labeler: done — {len(scorecards)} clusters in {elapsed:.1f}s, "
        f"{client.n_calls} LLM calls ({client.n_empty} empty)",
        "  confidence: " + "  ".join(f"{b}={counts[b]}" for b in bands if counts[b]),
        f"  flags: {n_sub} with sub-themes, {tally['confusable_pairs']} confusable pairs, "
        f"{tally['redifferentiated']} re-differentiated",
    ]
    if cohs:
        lines.append(f"  breadth: mean coherence {np.mean(cohs):.2f}, {n_low_coh} low-coherence clusters")
    return "\n".join(lines)


def labels_to_dataframe(scorecards: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for cid, sc in scorecards.items():
        sco = sc["scores"]
        b = sc.get("breadth") or {}
        rows.append({"cluster_id": cid, "label": sc["label"], "description": sc["description"],
                     "invariant_summary": b.get("invariant_summary", ""),
                     "varying_summary": b.get("varying_summary", ""),
                     "size": sc["size"], "recall": sco["recall"], "precision": sco["precision"],
                     "specificity": sco.get("specificity"), "discrimination": sco["discrimination"],
                     "stability": sco["stability"], "coherence": b.get("coherence"),
                     "confidence": sco["confidence"],
                     "n_invariant_axes": b.get("n_invariant", 0), "n_varying_axes": b.get("n_varying", 0),
                     "n_subthemes": len(sc["subthemes"] or []),
                     "n_confusable": len(sc["confusable_with"]), "note": sc["note"]})
    return pd.DataFrame(rows)


def render_label_report(scorecards: Dict[str, dict]) -> str:
    lines = ["=" * 70, f"CLUSTER LABELS  ({len(scorecards)} clusters)", "=" * 70]
    def _fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"

    for cid, sc in scorecards.items():
        sco = sc["scores"]
        conf = sco["confidence"].upper()
        lines.append(f"\n[{cid}] {sc['label']}  (size {sc['size']:,}, confidence {conf}, "
                     f"discrimination {_fmt(sco['discrimination'])}, stability {_fmt(sco['stability'])})")
        lines.append(f"   recall {_fmt(sco['recall'])}  precision {_fmt(sco.get('precision'))}  "
                     f"specificity {_fmt(sco.get('specificity'))}")
        if sc["description"]:
            lines.append(f"   {sc['description']}")
        b = sc.get("breadth") or {}
        if b.get("invariant_summary"):
            coh = f" (coherence {_fmt(b.get('coherence'))})" if b.get("coherence") is not None else ""
            lines.append(f"   shared: {b['invariant_summary']}{coh}")
        if b.get("varying_summary"):
            lines.append(f"   varies: {b['varying_summary']}")
        if sc["note"]:
            lines.append(f"   note: {sc['note']}")
        if sc["subthemes"]:
            for st in sc["subthemes"]:
                lines.append(f"   sub-theme: {st['name']} (size {st['size']})")
        if sc["confusable_with"]:
            for cw in sc["confusable_with"]:
                lines.append(f"   confusable with [{cw['cluster_id']}] {cw['label']} "
                             f"(centroid sim {cw['cent_similarity']})")
    return "\n".join(lines)
