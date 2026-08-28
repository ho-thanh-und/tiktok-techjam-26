# Autonomous ML Research Agent

This repository now includes a working, resumable autonomous ML control plane. Start with the [product guide](docs/product.md) and repository rules in [AGENTS.md](AGENTS.md).

```powershell
python -m automl_agent --config configs/demo.json preflight
python -m automl_agent --config configs/demo.json run --run-id demo-001
python -m unittest discover -s tests -v
python -m automl_agent --config configs/demo.json serve --port 8765
```

To let Gemini 3.7 Flash choose each catalog experiment, open the ignored root `.env` file and set:

```dotenv
GEMINI_API_KEY=your_real_key_here
```

Then use the LLM configuration:

```powershell
python -m automl_agent --config configs/demo-llm.json preflight
python -m automl_agent --config configs/demo-llm.json run --run-id llm-demo-001
```

The `.env` file is excluded from Git. The key is not copied into configuration or artifacts. Offline tests use local reasoning responses and do not call Gemini.

The demo exercises click-positive NDCG@10/Recall@50 orchestration end to end. The competition configuration intentionally fails closed until the organizer-provided baseline, evaluator, split, official selection score, and schema are supplied.

With the public KuaiRand-Pure files installed, `configs/kuairand-public-research.json` provides a real-data integration run. It is explicitly non-competition and cannot substitute for the organizer protocol.

## Legacy KuaiRand-Pure Starter Kit

The material below describes the original `long_view`/GAUC/nDCG@5 starter pipeline. It is retained for reference but does **not** match the current competition contract and must not be used as the official baseline.

> Planning an autonomous research system on top of this kit? Start with
> [architecture.md](architecture.md), then see the
> [companion system plan](docs/agent-ml-system-plan.md).
>
> For the recommender-system ML workflow, experiment ladder, and engineering
> checklists, see [ML.md](ML.md).

## Dependencies

Python 3.9+ and `numpy`. That's it — no `torch`, `pandas`, or `sklearn` required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# Run in the Starter Kit directory; this extracts to ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

The default `--data_dir` is `./KuaiRand-Pure/data`. If your data is elsewhere, specify it explicitly.

`--model` accepts `fm` (official baseline), `pop` (trivial popularity baseline), or `random` (lower bound, for harness sanity checks).
FM runs in about 40s on a single CPU core.

## Task definition (fixed — do not change)

|                           |                                                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Task                      | **Within-user ranking** — rank only a user's exposures in the evaluation set, not full-corpus retrieval                      |
| Relevance label           | `long_view` (native column, 0/1)                                                                                             |
| Metrics                   | `GAUC`, `nDCG@5`; **primary = mean(GAUC, nDCG@5)**                                                                           |
| Data splits               | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508`                                             |
| Users with zero positives | nDCG counted as 0.0 and included in the mean; GAUC counts only users with `0 < positives < exposures`, weighted by positives |
| nDCG gain                 | `2^rel − 1` (for binary labels this is equivalent to identity)                                                               |

Implementation is in `evaluate.py`; all conventions are documented in that file's header.

## Baseline leaderboard

Test-set scores. Your target to beat is the FM row.

|                                     | GAUC       | nDCG@5     | primary    |
| ----------------------------------- | ---------- | ---------- | ---------- |
| random (lower bound, harness check) | 0.4996     | 0.4511     | 0.4753     |
| item popularity (trivial)           | 0.6308     | 0.5121     | 0.5715     |
| **FM (official baseline)**          | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ Note: nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

|                                                           | Fraction  | Effect on metrics                                                 |
| --------------------------------------------------------- | --------- | ----------------------------------------------------------------- |
| All-negative users (no `long_view` in a user's exposures) | **27.1%** | nDCG is **0** for these users (irrecoverable); excluded from GAUC |
| All-positive users                                        | **9.2%**  | nDCG is **1** for these users; excluded from GAUC                 |
| Users with meaningful signal                              | **63.7%** | Actual sample for GAUC                                            |

Therefore, even an oracle that uses ground-truth labels as scores (perfect ranking) achieves:

|             | random | FM baseline | **oracle upper bound** | FM portion of available headroom |
| ----------- | ------ | ----------- | ---------------------- | -------------------------------- |
| GAUC        | 0.4996 | 0.6610      | **1.0000**             | 32.3%                            |
| nDCG@5      | 0.4511 | 0.5282      | **0.7289**             | 27.8%                            |
| **primary** | 0.4753 | **0.5946**  | **0.8645**             | **30.7%**                        |

Use the oracle as the denominator when reporting progress. Interpreting 0.5946 as far from 1.0 is misleading — the baseline already captures about 30% of the usable range; remaining headroom is ~0.27, not 0.41.

FM's standard deviation across 5 random seeds is **0.0008**. Based on this, we use a convergence rule of **ε = 0.002 (≈2.5σ), N = 3**:
if the primary validation score improves by no more than 0.002 for 3 consecutive iterations, consider the model converged.

Self-check: if `python3 baseline.py --model random` does not produce primary ≈ 0.475 (±0.001), your harness likely has a bug — fix it first.

## Submission format

CSV with a header; one row per evaluation-row:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field                  | Description                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `row_id`               | Zero-based consecutive index matching the row order returned by `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` then `log_standard_4_22_to_5_08_pure.csv`, apply date filter, keep original file order) |
| `user_id` / `video_id` | Redundant fields for alignment checks                                                                                                                                                                                                  |
| `score`                | Model score (any real number; only relative ordering matters). `NaN`/`Inf` not allowed                                                                                                                                                 |

Why `row_id` is required: `(user_id, video_id)` is not unique in the test set — there are 3.06% duplicate pairs (up to 12 repeats), so it can't serve as a primary key.

Generate and validate submissions:

```bash
python3 submit.py --make  --split test  submission.csv    # create an example submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score locally (valid split only)
```

`--check` will reject: incorrect header, wrong number of rows, missing/wrong `row_id` sequence, misaligned `user_id`/`video_id`, or non-numeric/NaN/Inf `score`. Run `--check` before submitting.

## Where to start improving

The list below is ordered by what the organizers actually tried; tested dead-ends are marked so you don't repeat them.

### Tested — no gain (do not waste iterations)

| Tried                                                                                                                      | Result                                                                           |
| -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Add static features** — add all 13 CWM feature domains (+ `music_id`/`video_type`/`upload_type` + 6 coarse user buckets) | primary **0.5940** vs 5-domain **0.5950** (no improvement, possibly slight drop) |
| **Increase model capacity** — embedding dimension k = 8 / 16 / 32                                                          | 0.5895 / 0.5902 / 0.5887 (negligible change)                                     |

Reason: the `user_id × video_id` cross already captures most learnable signal. Coarse user buckets like `follow_user_num_range` are redundant next to `user_id`, and 1.14M rows cannot support much larger capacity. **The bottleneck is not features or capacity.**

Note: **First-order user-only features contribute zero to ranking** because ranking is within-user; any user-constant term does not change intra-user order. User features can only help via cross terms with item-side features.

### Unexplored — likely places for headroom

These items were not tried by the organizers — good opportunities for you:

1. Change the loss. Current training is pointwise logloss, but metrics (GAUC / nDCG) are ranking metrics. Try pairwise (BPR) or listwise (softmax over a user's exposures) losses to align training and evaluation.
2. Use user history sequences. Current features ignore behavior sequences; users have hundreds to thousands of interactions in train. Methods like DIN/SASRec could help.
3. Multi-task learning. Logs include `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms` — useful auxiliary tasks to support `long_view`.
4. Model watch time. CWM models watch time via censored regression (watch time is truncated when a video ends), which suggests specialized loss functions.
5. Try different models: DeepFM / DCN / xDeepFM — lower priority after 1–4.
6. Time features and distribution shift: `hourmin`, `date`, and train/test drift.
7. Unbiased evaluation (advanced): `log_random_4_22_to_5_08_pure.csv` is 1.18M random-exposure logs and can serve as an unbiased validation set to detect bias-only improvements.

## Using your own model (including CWM)

`evaluate.py` is model-agnostic and accepts three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: user_id for each row in the evaluation set
- `labels`: the `long_view` label (0/1) for each row
- `scores`: model scores for each row (any real numbers; only relative ordering matters)

You can ignore `baseline.py` and use PyTorch, LightGBM, or CWM's xDeepFM — as long as you produce `scores` and pass them to `evaluate()`. **`evaluate.py` defines the official scoring protocol.**

Note on CWM: it requires `torch==1.6.0` (older GPU drivers may not support this), and it optimizes a counterfactual watch-time objective using a reconstructed `long_view2` label. Use CWM as an advanced reference, not a beginner-friendly starting point.

## Files

|                        |                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `evaluate.py`          | Metric implementation + all protocol conventions. **Do not modify.**                |
| `data.py`              | Data loading, official splits, feature encoding. Add new features here.             |
| `baseline.py`          | Three baselines; FM is the one to beat.                                             |
| `baseline_scores.json` | Official scores + seed variances + convergence parameters.                          |
| `submit.py`            | Create / validate submission files.                                                 |
| `ablation_features.py` | Feature ablation experiments; reproduces the "no gain from added features" numbers. |
| `knowledge/recommender-systems/` | Decision-routed recommender-system context for the ML agents.              |
