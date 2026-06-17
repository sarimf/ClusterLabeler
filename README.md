# cluster_labeler

Contrastive, **verified** semantic labeling for text clusters.

Give it a `DataFrame` of text + cluster ids (plus embeddings) and an LLM callable, and it
returns a per-cluster **scorecard**: a label, description, rationale, classifier-style
confidence scores, and the evidence behind every claim.

## Why not just "summarize the centroid neighbours"?

A label is a **decision boundary**, not a caption. `cluster_labeler` generates and grades each
label the way you'd build and test a classifier:

1. **Evidence** — core (typical) + diverse (sub-modes) + boundary (where this cluster blurs into
   its nearest neighbour) + contrast samples from neighbouring clusters ("what this is *not*").
2. **Decompose** — split the cluster into **invariant axes** (the attributes every member shares
   *and* that separate it from neighbours — its shared identity) and **varying axes** (the
   dimensions members differ on). Done *before* the label so it names the shared essence, not an
   incidental varying attribute.
3. **Propose** — several candidate cards, each required to be true of the target *and false of the
   neighbours* (contrastive, not just descriptive), guided by the invariant/varying axes.
4. **Verify** — score each candidate as a classifier on **held-out** items it never saw:
   recall, precision, specificity, and discrimination (balanced accuracy). The invariant axes are
   also checked against held-out members to yield a **coherence** score (low = the cluster lacks a
   real shared identity).
5. **Refine** — feed the best candidate's false negatives/positives back, ask for a revision,
   repeat until it clears the bar or the iteration budget runs out.
6. **Stability** — resample the evidence and re-propose; a label that survives resampling is
   trustworthy, one that flips means the cluster itself is ill-defined.
7. **Sub-themes + global coherence** — flag clusters that look like two things, and flag pairs of
   clusters whose labels collide (re-differentiated) or whose content overlaps (flagged for merge
   review).

Every claim a label makes is backed by evidence a human can re-check; every score comes from
held-out items the candidate never saw.

---

## Install

```bash
pip install numpy pandas scikit-learn
pip install tqdm        # optional, for a progress bar
```

Then drop `cluster_labeler.py` into your project (single-file module).

---

## Minimal version

You need three things: a `DataFrame` (text + cluster id), an aligned embeddings matrix, and an
LLM gateway. The gateway is any callable `fn(messages, json_mode=True) -> str` that returns the
model's raw text.

```python
import pandas as pd
import numpy as np
from cluster_labeler import use_llm, label_clusters, render_label_report

# 1. your data: one row per item, with a cluster assignment
df = pd.DataFrame({
    "text":       ["card declined at checkout", "refund still not received", ...],
    "cluster_id": [3, 3, ...],
})
embeddings = np.load("embeddings.npy")   # shape (len(df), d), row-aligned to df

# 2. register an LLM gateway once (example: OpenAI-style chat call)
def my_llm(messages, json_mode=True):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"} if json_mode else None,
    )
    return resp.choices[0].message.content

use_llm(my_llm)            # or as a decorator: @use_llm above `def my_llm(...)`

# 3. label every cluster
scorecards = label_clusters(df, embeddings=embeddings)
print(render_label_report(scorecards))
```

> The gateway is called from multiple worker threads concurrently — make sure `my_llm` (and any
> client it wraps) is thread-safe, or lower `workers` (see below).
>
> **If a run seems to hang:** it's almost always a stalled gateway call (rate-limit, dropped
> connection, no server-side timeout). Each call is bounded by `request_timeout` (default 60s) so a
> single hung request can't freeze the batch — lower it (e.g. `LabelConfig(request_timeout=20)`)
> for snappier failure, and/or set a timeout inside `my_llm` itself.
>
> **If your gateway already retries/backs off** (e.g. `tenacity`, or the OpenAI client's built-in
> retries), don't let the two layers fight: a long backoff inside `my_llm` can exceed
> `request_timeout`, which then cancels the call mid-backoff and looks like a hang. Set
> **`request_timeout=0`** (rely on your gateway's own per-request timeout) and **`max_retries=0`**
> (your gateway owns retries), and consider lowering `workers` (e.g. 4–8) so you stop hitting the
> rate limits that triggered the backoff in the first place. (A timeout is never retried by
> `cluster_labeler` anyway — it's terminal — but the layers can still waste a lot of wall-clock.)

### No model handy? Offline mock (testing/demos only)

If you don't register a gateway, labeling **raises** by default so a misconfigured run can't
silently emit junk. To use the built-in offline mock (a crude word-overlap heuristic — *not*
model quality), opt in explicitly:

```python
from cluster_labeler import LabelConfig, label_clusters
cards = label_clusters(df, embeddings=embeddings, cfg=LabelConfig(allow_mock=True))
```

---

## Thorough version

```python
import pandas as pd
import numpy as np
from cluster_labeler import (
    LabelConfig, use_llm, label_clusters, labels_to_dataframe, render_label_report,
)

use_llm(my_llm)

cfg = LabelConfig(
    # tell the model what it's looking at — improves labels a lot
    domain_hint="customer-support chat messages",
    same_when="they describe the same underlying issue",

    # spend more on evidence + candidates for higher-stakes labeling
    n_core=10, n_diverse=8, n_candidates=6, refine_max_iters=3,

    # stricter acceptance bars
    accept_discrimination=0.85, accept_recall=0.75, accept_precision=0.75,

    # judge / runtime
    model="gpt-4o", temperature=0.1, seed=7, workers=8,
)

scorecards = label_clusters(
    df,
    embeddings=embeddings,        # OR: embedding_col="emb" to read vectors from a df column
    text_col="text",
    cluster_col="cluster_id",
    cfg=cfg,
    progress=True,
)

# render / export
print(render_label_report(scorecards))
summary = labels_to_dataframe(scorecards)        # one tidy row per cluster
summary.to_csv("cluster_labels.csv", index=False)

# drill into one cluster
card = scorecards["3"]
print(card["label"], card["scores"]["confidence"])
for txt in card["evidence"]["core_texts"]:        # the exemplars behind the label
    print("  -", txt)
```

### Passing embeddings via a DataFrame column

```python
# df["emb"] holds an equal-length vector per row
scorecards = label_clusters(df, embedding_col="emb", cfg=LabelConfig(...))
```

### Per-call override of the gateway

`use_llm(...)` registers a process-wide default; you can also pass one per call, which takes
precedence:

```python
scorecards = label_clusters(df, embeddings=embeddings, llm_fn=my_llm)
```

---

## Watching it run (progress & logging)

By default (`verbose=1`) the run is **not** a black box — it prints a banner, one line per
finished cluster, any coherence flags, and a final summary to stderr (regardless of your logging
setup):

```text
cluster_labeler: labeling 4 clusters over 109 items (model=gpt-4o, workers=8)
  cluster sizes: min 4, median 32, max 40; acceptance bars: disc≥0.8 rec≥0.7 spec≥0.7
✓ [1/4] [billing] 'Payment & refund disputes'  size=40  HIGH   disc=0.95 rec=0.97 prec=1.00 spec=0.92 stab=0.61  calls=7
• [2/4] [login]   'Account access / password resets'  size=35  MEDIUM  disc=0.78 ...
? [3/4] [tiny]    'Misc edge cases'  size=4  UNVERIFIED  ...  (cluster too small for held-out verification)
  ⚠ confusable content: [shipping] 'Delivery delays' ~ [returns] 'Return delays' (centroid sim 0.61)
cluster_labeler: done — 4 clusters in 18.3s, 612 LLM calls (3 empty)
  confidence: high=1  medium=1  unverified=1  error=1
  flags: 0 with sub-themes, 1 confusable pairs, 0 re-differentiated
```

- `verbose=0` — silent.
- `verbose=1` — banner + per-cluster result + coherence flags + summary (default).
- `verbose=2` — also per-stage detail inside each cluster (evidence shape, candidates proposed,
  each candidate's grades, sub-themes).

Even though clusters are labeled in parallel (`workers`), each cluster's lines are **buffered and
flushed as one block** when it finishes, so the output stays grouped and readable instead of
interleaving across workers.

### Progress bar

With `tqdm` installed and `progress=True` (default), a single live bar tracks clusters finished,
with the running **LLM-call count and rate in its postfix**:

```text
labeling:  75%|███████▌  | 3/4 [00:12<00:04, cluster/s, 612 llm calls · 49.7/s]
```

The call count rides in the postfix (rather than a second bar — a second stacked bar spams a new
line per update inside Jupyter) and is handy for spotting a stalled gateway: the calls/sec drops to
zero. `tqdm.auto` is used, so you get the native widget bar in notebooks and the console bar
elsewhere. The bar is independent of `verbose`; set `progress=False` to disable it.

Every message is also sent to the `cluster_labeler` logger, so apps that configure logging can
capture or route it; set `verbose=0` and use logging if you prefer.

## What you get back

`label_clusters(...)` returns `{cluster_id: scorecard}`. Each **scorecard** is:

| Field | Meaning |
|---|---|
| `cluster_id` | The cluster id (as a string). |
| `label` | Short contrastive label (≤ ~8 words). |
| `description` | One-line description (clipped to `desc_chars`). |
| `rationale` | What separates this cluster from its neighbours. |
| `size` | Number of items in the cluster. |
| `scores` | `recall`, `precision`, `specificity`, `discrimination`, `stability`, `confidence`. |
| `alternatives` | Runner-up candidate cards with their discrimination scores. |
| `confusable_with` | Other clusters with near-identical content (merge-review candidates). |
| `breadth` | The axis decomposition: `summary`, `invariant_axes`, `varying_axes`, `coherence`, counts (see below). |
| `subthemes` | If the cluster looks like two things, the named sub-groups (informational). |
| `evidence` | `core` / `diverse` / `boundary` row indices **and** their `*_texts`. |
| `n_llm_calls` | LLM calls spent on this cluster. |
| `note` | Caveats (e.g. "cluster too small for held-out verification"). |

**Score meanings** (all from held-out items the candidate never saw):

- `recall` — share of held-out members the label accepts. `TP / (TP+FN)`
- `precision` — of the items it accepts, the share that are true members. `TP / (TP+FP)`
- `specificity` — share of sibling items it correctly rejects. `TN / (TN+FP)`
- `discrimination` — balanced accuracy of recall + specificity.
- `stability` — token overlap of labels across evidence resamples (`None` = not assessed).
- `confidence` — `high` / `medium` / `low` / `unverified` / `error`. **`high` requires** every
  bar to clear *and* specificity to have been measured (a cluster with no siblings to reject
  cannot earn `high`).

### Breadth — axes of variation vs invariance

`breadth` factors each cluster into *what stays constant* vs *what changes*:

- `summary` — a prose description of the range the cluster spans (like a `description`, for breadth).
- `invariant_axes` — `[{axis, value}]` shared by (nearly) every member **and** distinctive vs
  neighbours. This is the cluster's identity, and what the label is steered to name.
- `varying_axes` — `[{axis, values[], open_ended, example_ids}]` the dimensions members differ on,
  with the observed value range (`open_ended=true` means the list is illustrative, not exhaustive).
- `coherence` — share of held-out members consistent with **all** invariant axes (`None` when not
  verified / no holdout). Low coherence ⇒ the cluster mixes things that don't share an identity — a
  quality signal complementing `discrimination`.
- `n_invariant`, `n_varying` — counts.

The decomposition is computed **before** the label (so the label names the shared identity, not an
incidental varying attribute) and costs ≈1–2 extra LLM calls per cluster.

`labels_to_dataframe(scorecards)` flattens this to one row per cluster with columns:
`cluster_id, label, description, breadth_summary, size, recall, precision, specificity,
discrimination, stability, coherence, confidence, n_invariant_axes, n_varying_axes, varying_axes,
n_subthemes, n_confusable, note`.

---

## All tunable parameters (`LabelConfig`)

### Prompt context
| Param | Default | What it does |
|---|---|---|
| `domain_hint` | `None` | Free-text description of the corpus, injected into every prompt. Strongly recommended. |
| `same_when` | `None` | Your definition of "same kind" (the merge rule), injected into prompts. |
| `item_chars` | `400` | Max characters per item shown to the model (longer items are clipped). |
| `desc_chars` | `160` | Max characters kept for each generated description. |

### Evidence shape
| Param | Default | What it does |
|---|---|---|
| `n_core` | `8` | Nearest-centroid exemplars (the "typical" member). |
| `n_diverse` | `6` | Micro-mode medoids capturing the spread / sub-themes. |
| `n_boundary` | `4` | Items closest to the nearest sibling centroid (where the cluster blurs). |
| `micro_k` | `6` | Number of micro-modes computed for the diverse sample. |
| `n_contrast_clusters` | `3` | Nearest sibling clusters shown as "what this is *not*". |
| `n_contrast_items` | `4` | Exemplars shown per contrast cluster. |

### Held-out verification
| Param | Default | What it does |
|---|---|---|
| `holdout_frac` | `0.3` | Fraction of a cluster's members held out for grading (never shown to evidence/proposal/refine). |
| `min_holdout` | `4` | Minimum held-out members (when the cluster is large enough). |
| `verify_positives` | `12` | Held-out members **sampled** into each verify prompt. Caps prompt size on big clusters (the full holdout can be thousands of items); raise for a tighter recall estimate, lower for cheaper/faster calls. |
| `verify_negatives` | `8` | Held-out sibling items used as negatives during grading. |

### Candidate generation & acceptance
| Param | Default | What it does |
|---|---|---|
| `n_candidates` | `4` | Distinct candidate cards proposed per cluster. |
| `refine_max_iters` | `2` | Max refine rounds for the best candidate. |
| `accept_discrimination` | `0.80` | Discrimination bar for "high" confidence and for stopping refinement. |
| `accept_recall` | `0.70` | Recall bar required for "high" confidence. |
| `accept_precision` | `0.70` | Specificity bar required for "high" confidence. |

### Stability
| Param | Default | What it does |
|---|---|---|
| `stability_resamples` | `2` | Evidence resamples used to test label stability (`0` disables → stability `None`). |
| `stability_min_jaccard` | `0.34` | Min label-token overlap across resamples to count as stable for "high". |

### Breadth (axes of variation vs invariance)
| Param | Default | What it does |
|---|---|---|
| `breadth_exemplars` | `14` | Diverse target exemplars shown to the decomposer (capped for prompt size). |
| `breadth_max_axes` | `8` | Cap on the number of varying axes listed. |
| `breadth_resamples` | `1` | `>1` unions independent extractions on resampled evidence (higher axis recall, proportional cost). |
| `breadth_verify` | `True` | Verify invariant axes on held-out members to produce `coherence`. |

### Sub-theme detection (informational; superseded by breadth, retained for compat)
| Param | Default | What it does |
|---|---|---|
| `subtheme_spread_ratio` | `1.6` | Inter/intra micro-mode spread that triggers sub-theme splitting. |
| `subtheme_min_size` | `16` | Minimum cluster size before sub-themes are considered. |

### Global coherence pass
| Param | Default | What it does |
|---|---|---|
| `dedup_label_jaccard` | `0.6` | Label-text overlap above which two clusters are re-differentiated. |
| `dedup_cent_sim` | `0.55` | Centroid similarity above which two clusters are flagged as confusable / merge-review. |

### Small clusters
| Param | Default | What it does |
|---|---|---|
| `min_cluster_size` | `6` | Below this, skip the held-out split, label from all members, and mark low confidence. |

### Judge / runtime
| Param | Default | What it does |
|---|---|---|
| `model` | `"mock"` | Model id (metadata for your gateway; the gateway decides the actual model). |
| `temperature` | `0.2` | Sampling temperature (metadata for your gateway). |
| `seed` | `7` | RNG seed — labeling is **deterministic** for a fixed seed + inputs + gateway. |
| `workers` | `16` | Thread-pool size. Your `llm_fn` must be thread-safe; lower this if not. |
| `max_retries` | `3` | Retries per LLM call on error/unparseable output (exponential backoff). |
| `backoff_base` | `0.5` | Initial backoff seconds between retries (doubles each retry). |
| `request_timeout` | `60.0` | Seconds to wait for each gateway call before giving up (then retried). Bounds a stalled request so one hung call can't hang the whole batch. `0`/`None` disables. |
| `allow_mock` | `False` | Must be `True` to run the offline mock when no gateway is registered. |

> **Note:** `model` and `temperature` are stored on the config but not currently passed into the
> `fn(messages, json_mode)` gateway — your gateway chooses the real model/temperature. Wire them
> in yourself if you need per-call control.

---

## API reference

```python
use_llm(fn)            # register a process-wide gateway: fn(messages, json_mode=True) -> str
use_genai(fn)          # alias of use_llm

label_clusters(
    df,                          # DataFrame with text_col + cluster_col
    embeddings=None,             # (n_rows, d) array aligned to df ...
    embedding_col=None,          # ... OR a df column holding per-row vectors
    text_col="text",
    cluster_col="cluster_id",
    cfg=None,                    # LabelConfig (defaults if omitted)
    llm_fn=None,                 # per-call gateway (overrides use_llm)
    progress=True,               # tqdm progress bar (if tqdm installed)
    verbose=1,                   # 0 silent | 1 per-cluster + summary | 2 per-stage detail
) -> dict[str, scorecard]

labels_to_dataframe(scorecards) -> pd.DataFrame    # one tidy row per cluster
render_label_report(scorecards) -> str             # human-readable text report
```

### Input requirements
- `df` is non-empty and contains `text_col` and `cluster_col`.
- Embeddings are 2-D, row-aligned to `df`, and free of `NaN`/`inf` (validated; clear errors otherwise).
- A gateway is registered (`use_llm` / `llm_fn`) **or** `LabelConfig(allow_mock=True)` is set.

---

## Testing

```bash
python test_cluster_labeler.py      # offline, runs against the built-in mock (no network/key)
```
