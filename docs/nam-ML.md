# Recommender-System ML Engineering Workflow

Status: practical workflow and research guide  
Repository target: KuaiRand-Pure autonomous ML research agent  
Competition task: rank each user's logged impressions for `long_view`

For the decision-by-decision agent tuning loop, see the
[detailed ML flowchart and agent decision map](nam-ml-agent-tuning-flow.md).
For decision-routed recommender context, see the
[recommender-systems knowledge index](../knowledge/recommender-systems/README.md).

## 1. What a recommender-system ML engineer does

A recommender-system ML engineer turns user behavior into a reliable ranking system. The work is much broader than choosing a neural-network architecture.

Normally, the engineer:

1. Defines the recommendation surface, user action, prediction time, label, and success metric.
2. Audits impression and feedback logs to understand what was shown, under which policy, and what the user did afterward.
3. Builds reproducible train/validation/test datasets without future or target leakage.
4. Establishes simple heuristic and ML baselines.
5. Creates user, item, context, aggregate, and sequence features.
6. Trains retrieval and/or ranking models with objectives aligned to the product task.
7. Evaluates overall quality, important user/item segments, stability, bias, diversity, latency, and resource cost.
8. Runs controlled experiments and keeps only changes supported by evidence.
9. Packages the winning feature pipeline and model for inference.
10. Monitors data freshness, training-serving skew, model quality, latency, and feedback loops after deployment.

In practice, most engineering effort goes into data, evaluation, experimentation, serving, and monitoring rather than the model class itself. Google's production ML guidance similarly treats model code as a small part of the overall system and recommends trustworthy infrastructure and simple models before additional complexity ([Production ML Systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems), [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)).

## 2. Understand which recommendation problem you are solving

Production recommenders commonly form a funnel:

```text
Eligible catalogue
      ↓
Candidate retrieval       millions → hundreds/thousands
      ↓
Pre-ranking               thousands → hundreds
      ↓
Ranking                   hundreds → tens
      ↓
Re-ranking/policy         final ordered list
      ↓
User interaction and new logs
```

Google's recommender overview describes candidate generation, scoring, and re-ranking as distinct stages. Large-scale systems such as YouTube also use separate candidate-generation and ranking models ([Google recommendation overview](https://developers.google.com/machine-learning/recommendation/overview/types), [YouTube recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)).

### This repository's scope

KuaiRand-Pure in this challenge is **not a full-catalogue retrieval problem**. The evaluation already provides the impressions to rank. The model only assigns a score to each logged `(user, video, context)` row and ordering is measured within each user.

Therefore:

- Spend the competition budget on ranking, representation, features, losses, and evaluation.
- Do not build ANN retrieval for the primary benchmark; it cannot change the official logged-impression metric.
- Retrieval remains relevant knowledge for a future production deployment or the bonus datasets if their task changes.

## 3. Start with a written problem contract

Before opening a notebook, write down:

| Question | Example for this repository |
| --- | --- |
| What is scored? | Order of logged impressions within each user |
| What is one example? | A user–video impression at a particular time/context |
| What is known at prediction time? | User/item IDs, static attributes, context, permitted past behavior |
| What is unknown? | The current row's `long_view` and other future feedback |
| Primary label | Native binary `long_view` |
| Primary metrics | GAUC and nDCG@5; equal-weighted mean |
| Split strategy | Fixed chronological train, validation, and hidden test |
| Candidate set | Evaluation rows already logged for each user |
| Baseline to beat | Official FM, validation primary `0.6016` |
| Resource limits | 50 iterations and six-hour wall-clock |

This contract prevents a common failure: optimizing a technically interesting task that is different from the task the evaluator or product actually measures.

## 4. Data and logging workflow

### 4.1 Know the event model

A useful production impression record usually needs:

- Event and request IDs.
- User/account ID or privacy-safe equivalent.
- Candidate item ID.
- Impression timestamp.
- Surface, device, locale, and request context.
- Position shown.
- Retrieval source and serving policy/model version.
- Model score and other candidates considered.
- Action propensity if recommendations include randomized exploration.
- Feedback events with timestamps: click, watch time, long view, like, follow, comment, forward, hide, report, and so on.

Logging only positive interactions is insufficient. The engineer needs impressions to know what the user could have selected and to construct meaningful negatives.

KuaiRand is particularly useful because it includes sequential logs, rich side information, 12 feedback signals, and randomized exposures. The randomized data supports investigation of exposure bias and off-policy evaluation ([KuaiRand project](https://kuairand.com/), [KuaiRand paper](https://arxiv.org/abs/2208.08696)).

### 4.2 Validate the raw data

Run deterministic checks before EDA or training:

- Required files and columns exist.
- IDs, timestamps, labels, and numeric ranges are valid.
- Row counts and split boundaries match the benchmark contract.
- Duplicate rows and repeated user–item pairs are quantified, not silently removed.
- Label prevalence is measured overall and by date, user, item, tab, and duration.
- Missing/unknown feature rates are reported per split.
- Items and users unseen in training are measured in validation.
- Timestamps are ordered correctly and expressed in one timezone.
- No label or post-outcome field is present in inference features.

Fail fast if these checks change between experiments. Otherwise an apparent model improvement may only be a data-pipeline change.

### 4.3 Split by time

Random row splits allow future behavior to leak into the past and usually make recommender performance look unrealistically strong. Prefer chronological splits that match deployment:

```text
past                      prediction future
|--------- train ---------|--- validation ---|--- test ---|
```

For repeated development, create rolling folds inside the training period:

```text
fold 1: early train → later dev
fold 2: more train  → later dev
fold 3: most train  → latest internal dev
```

Use internal folds for cheap screening and the official validation split for comparable experiment decisions. Never tune against hidden test.

### 4.4 Prevent target and future leakage

For every feature, ask: **Could this exact value exist at the moment the recommendation is scored?**

Rules:

- Fit vocabularies, normalizers, popularity, and aggregate statistics on training history only.
- For a training row at time `t`, calculate historical features using events strictly before `t`, or use out-of-fold construction.
- Do not use the current impression's click, watch time, like, or `long_view` as an input.
- Auxiliary feedback may be a training target, but it remains unavailable for the same row during inference.
- Audit precomputed item-statistics files: a statistic built using the full logging period may leak validation/test outcomes.
- Use one feature implementation for training and inference where possible.

Google's production guidance recommends using only features available at prediction time and explicitly monitoring schema and feature skew between training and serving ([Monitoring ML pipelines](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)).

## 5. Exploratory data analysis

EDA should answer decisions, not merely generate charts.

### 5.1 Dataset shape

Measure:

- Interactions, users, items, authors, and dates.
- Interactions per user and item, using median and percentiles as well as mean.
- Positive-label rate and feedback-funnel rates.
- Number of evaluation impressions and positives per user.
- Long-tail concentration: percentage of interactions owned by the top 1%, 5%, and 10% of items.
- Cold-start rates for users, items, and authors.

### 5.2 Temporal behavior

Measure by date/hour:

- Traffic volume.
- Label and feedback rates.
- Item/author popularity.
- New and disappearing items.
- Feature missingness.
- Train-to-validation distribution changes.

These results determine whether recency weighting, time features, rolling training windows, or special cold-start handling are justified.

### 5.3 Bias and exposure

Logged recommendations reflect the previous serving policy. Popular items and high positions receive more observations, so the data is not an independent sample of all user preferences.

Measure:

- Label rate by original position, tab, and serving policy.
- Standard-log versus randomized-exposure distributions.
- Propensity coverage and very small propensities.
- Which users/items rarely receive exposure.

Randomized or propensity-logged data can support replay, inverse propensity scoring, doubly robust methods, or counterfactual risk analysis. These methods require explicit assumptions and variance checks; they should not be applied as a cosmetic weight. Original logged-bandit work explains why observed feedback is partial and policy-biased and why importance weighting needs support from the logging policy ([Unbiased offline evaluation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf), [Deep learning with logged bandit feedback](https://www.microsoft.com/en-us/research/publication/deep-learning-logged-bandit-feedback/)).

## 6. Establish baselines before complex models

Run several rungs through exactly the same pipeline:

1. Random scores: scorer and alignment sanity check.
2. Global popularity: verifies labels and item aggregation.
3. Smoothed item/author popularity: stronger non-personalized baseline.
4. Matrix factorization or FM: personalized ID-interaction baseline.
5. A strong tabular ranker: validates engineered features.

For each baseline, save:

- Configuration and seed.
- Data/source/environment hashes.
- Training curves and runtime.
- Validation predictions and official metrics.
- Segment metrics.
- Checkpoint and submission-format validation.

Simple baselines reveal whether expensive models are learning real personalization or reproducing popularity. They also make infrastructure bugs easier to diagnose. Google's Rules of ML explicitly recommends starting with solid metrics, infrastructure, heuristics, and simple models before adding complexity ([Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)).

## 7. Feature engineering

### 7.1 User features

- User ID embedding.
- Activity count and active days.
- Historical click/long-view/like rates with smoothing.
- Preferred duration, author, category, and tab distributions.
- Time since last interaction.
- Recent-session length and activity intensity.

A user-only additive score does not change within-user ranking. User features help this task when they interact with item, author, context, or history features.

### 7.2 Item and author features

- Video and author ID embeddings.
- Duration and duration bucket.
- Content/category/music/upload attributes.
- Age/freshness at impression time.
- Smoothed historical exposure and positive rates.
- Trend velocity over recent windows.
- Cold-start/unknown indicators.

### 7.3 Cross and affinity features

- User × item and user × author interaction.
- User preference versus candidate category/duration.
- Count and recency of prior user interactions with the item or author.
- Similarity between recent user-history embedding and candidate embedding.
- Candidate popularity relative to the user's normal consumption.

FM automates low-order feature interactions; DeepFM combines FM-style low-order interactions with a neural component for higher-order interactions ([DeepFM paper](https://arxiv.org/abs/1703.04247)). Do not assume a more complex interaction model is better—require an ablation.

### 7.4 Sequence features

Create histories in strict timestamp order and retain masks and sequence lengths. Useful sequence elements include:

- Recent video IDs, authors, categories, and durations.
- Feedback type and strength.
- Time gaps.
- Session boundaries.
- Candidate-conditioned similarity to prior behavior.

DIN constructs candidate-specific attention over user history, while SASRec uses self-attention to model which prior actions matter for a next-item decision ([DIN](https://arxiv.org/abs/1706.06978), [SASRec](https://arxiv.org/abs/1808.09781)). For KuaiRand, start with a bounded DIN-style history before a larger sequential model because the challenge already supplies candidate impressions and has a strict iteration budget.

### 7.5 Feature acceptance checklist

A feature is accepted only if:

- It is available at inference time.
- Its timestamp and construction window are documented.
- The same implementation is used for train/validation/test.
- Unknown and missing values have explicit behavior.
- An ablation shows useful validation gain or segment robustness.
- Its memory, training, and inference cost are justified.

## 8. Choose an objective that matches the task

### 8.1 Pointwise objectives

Binary cross-entropy treats every impression independently:

```text
model(user, item, context) → P(long_view = 1)
```

Advantages:

- Simple, stable, and compatible with auxiliary binary tasks.
- Produces interpretable probabilities when calibrated.

Limitation:

- It does not directly express that one item should rank above another for the same user.

Keep pointwise FM/logistic loss as a baseline, not the only objective.

### 8.2 Pairwise objectives

For a user's positive item `i` and negative item `j`, optimize:

```text
score(user, i) > score(user, j)
```

BPR is a standard pairwise objective for implicit-feedback personalized ranking and was explicitly designed to optimize item order rather than independent classification ([BPR paper](https://arxiv.org/abs/1205.2618)).

Best practices:

- Construct pairs inside the same user or request context.
- Ensure a user has both positive and negative examples.
- Log how negatives are sampled.
- Compare uniform, popularity-aware, and hard-negative sampling.
- Prevent a small number of highly active users from dominating all pairs.

### 8.3 Listwise and LambdaRank objectives

Listwise training treats a user's candidate group as an ordered list. LambdaRank/LambdaMART weights pair changes by their effect on ranking quality, which can align better with nDCG.

LightGBM provides `lambdarank` and `rank_xendcg`. Its documentation recommends relating `lambdarank_truncation_level` to the target NDCG cutoff; for nDCG@5, begin near `k + 3`, then validate ([LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).

Best practices:

- Sort rows by group and pass exact group sizes.
- Never allow one user/request to cross train/validation boundaries.
- Set the evaluation cutoff to the actual product or benchmark cutoff.
- Evaluate GAUC as well as nDCG; optimizing one can trade against the other.

### 8.4 Multi-task objectives

Rich-feedback recommenders often share representations across objectives such as click, long view, like, follow, and watch time. Joint training can transfer information from abundant tasks to sparse tasks, but task weights must be tuned and negative transfer monitored ([TensorFlow Recommenders multi-task guide](https://www.tensorflow.org/recommenders/examples/multitask/)).

A practical starting loss is:

```text
L = 1.0 * L_long_view
  + w_click * L_click
  + w_like * L_like
  + w_follow * L_follow
  + w_watch * L_watch_time
```

Start with shared embeddings plus separate heads. If tasks interfere, try task-specific towers or MMoE, which learns task-specific gates over shared experts and was evaluated on large content recommendation ([MMoE paper](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)).

Do not apply ESMM mechanically here. ESMM addresses a post-click conversion label observed only on clicked impressions. KuaiRand's scored `long_view` is recorded on every impression, so the classic ESMM sample-selection setup does not directly match this benchmark, although its representation-sharing idea remains relevant ([ESMM paper](https://arxiv.org/abs/1804.07931)).

### 8.5 Watch-time objectives

Raw watch time is influenced by video duration. A completed video can make observed watch time effectively censored at its duration. If watch time is used as an auxiliary target, compare normalized completion, robust regression, and censored objectives. CWM specifically studies duration bias and censored watch-time regression in video recommendation ([CWM paper](https://arxiv.org/abs/2406.07932)).

## 9. Model ladder

Move up this ladder only when the previous level is understood:

| Level | Candidate models | Purpose |
| --- | --- | --- |
| 0 | Random, popularity, recency | Pipeline checks and non-personalized lower bounds |
| 1 | Logistic regression, MF, FM | Strong interpretable ID/cross baseline |
| 2 | LightGBM LambdaRank/XENDCG | Fast rank-aware tabular baseline |
| 3 | DeepFM, DCN-style ranker | Higher-order feature interactions |
| 4 | Shared-bottom or MMoE multi-task ranker | Use multiple feedback signals |
| 5 | DIN | Candidate-aware behavior history |
| 6 | SASRec/sequence transformer | Long sequential preferences |
| 7 | Ensembles | Combine models with different errors |

For this repository, the recommended first serious challenger is a leakage-safe tabular ranker with a rank-aware objective, followed by BPR-FM and a small multi-task history-aware model. This gives evidence quickly before investing in sequence-model infrastructure.

## 10. Training workflow

For every experiment:

1. State one hypothesis and a measurable success criterion.
2. Freeze dataset, feature, code, environment, and seed manifests.
3. Fit preprocessing on training data only.
4. Train with logged loss, learning rate, gradient/weight norms, throughput, and memory.
5. Save recoverable checkpoints.
6. Generate finite, aligned validation scores.
7. Run the unchanged official evaluator.
8. Compare with both the official baseline and the current incumbent.
9. Run segment and error analyses.
10. Save the configuration, diff, metrics, artifacts, duration, and conclusion.

### Reproducibility controls

- Seed Python, NumPy, and the ML framework.
- Record nondeterministic accelerator operations when they cannot be disabled.
- Use a locked dependency environment.
- Store exact feature and data hashes.
- Run promising candidates over multiple seeds.
- Report mean and dispersion, not only the best seed.

### Hyperparameter tuning

- Tune a few high-impact parameters based on a hypothesis.
- Use coarse-to-fine or Bayesian search, not a blind large grid.
- Include runtime and memory in the search result.
- Avoid repeatedly querying the official validation split for tiny changes.
- Re-run the final candidates with multiple seeds before selecting an ensemble.

## 11. Evaluation workflow

### 11.1 Match metrics to the stage

| Stage | Typical offline metrics |
| --- | --- |
| Retrieval | Recall@K, hit rate, candidate coverage, ANN latency |
| Ranking | AUC/GAUC, nDCG@K, MAP, log loss, calibration |
| Re-ranking | Diversity, novelty, freshness, constraint violations, list utility |
| Production | Engagement/satisfaction, retention, complaints, latency, cost |

This challenge uses GAUC and nDCG@5 only. Do not optimize Recall@50 or full-catalogue retrieval unless the task contract changes.

### 11.2 Understand the official metrics

- **GAUC:** computes AUC within each eligible user and aggregates it. It tests whether positives outrank negatives for the same user.
- **nDCG@5:** rewards placing positives near the top of each user's list, with stronger weight on early positions.
- **Primary:** the equal-weighted mean of the two.

Track GAUC and nDCG separately. A model can improve the primary mean by helping one while hurting the other.

### 11.3 Always add segment evaluation

Report metrics for:

- Head versus tail items.
- Warm versus cold users/items/authors.
- Low/high user activity.
- Short/medium/long videos.
- Tab and time buckets.
- Users with small/large candidate groups.
- Train-known versus unseen categories.

Also inspect the largest positive and negative score changes between incumbent and candidate. A scalar average can hide systematic regressions; ranking-evaluation research similarly warns that one aggregate metric can obscure different user/list behaviors ([Offline retrieval evaluation](https://research.google/pubs/offline-retrieval-evaluation-without-evaluation-metrics/)).

### 11.4 Quantify uncertainty

- Repeat stochastic models over several seeds.
- Bootstrap users rather than individual rows because rankings are grouped by user.
- Report confidence intervals or at least standard deviation.
- Treat changes smaller than known run-to-run noise as uncertain.
- Use the challenge's `epsilon = 0.002` rule for convergence, not per-epoch early stopping.

### 11.5 Offline is not online

Logged metrics measure performance under a historical exposure policy. A production system also affects which future data it will observe. When online deployment exists, use staged rollout and randomized A/B tests with product and guardrail metrics. Accuracy alone does not cover diversity, novelty, safety, long-term satisfaction, or ecosystem effects.

## 12. Error analysis and reflection

After every useful experiment, answer:

1. Which metric and user/item segments changed?
2. Does the evidence support the original hypothesis?
3. Is the change larger than random-seed noise?
4. Could leakage, misalignment, or a data change explain the gain?
5. Did training time, memory, or inference cost increase?
6. What did this teach us about the next experiment?
7. Should the candidate be accepted, rejected, revised, or ensembled?

A failed hypothesis is valuable if its evidence is recorded. Do not repeat the same dead end under a slightly different name.

## 13. Serving and production workflow

This hackathon produces batch scores, but a production recommender engineer would also own or collaborate on:

### Retrieval serving

- Precompute candidate/item embeddings.
- Maintain an approximate nearest-neighbor index.
- Generate candidates from multiple sources: personalized, trending, subscriptions, fresh content, and exploration.
- Measure recall and latency by source.

TensorFlow Recommenders' retrieval documentation describes the standard two-tower query/candidate structure and ANN serving trade-off ([TFRS retrieval guide](https://www.tensorflow.org/recommenders/examples/basic_retrieval)).

### Ranking serving

- Reuse exactly the same feature definitions as training.
- Batch candidate scoring where possible.
- Define latency, memory, and model-size budgets.
- Handle missing and unseen features explicitly.
- Version model, feature schema, and calibration together.

### Re-ranking

Apply transparent policy constraints after model scoring:

- Remove ineligible or unsafe content.
- Limit repeated authors/topics.
- Balance relevance with diversity, freshness, and exploration.
- Enforce business and user-control rules.
- Log both pre- and post-policy scores.

Google's re-ranking guidance describes using the final stage to incorporate freshness, diversity, and other constraints beyond the ranking model ([Re-ranking guide](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking)).

### Rollout

1. Offline acceptance tests.
2. Shadow scoring without affecting users.
3. Small canary traffic.
4. Randomized A/B test.
5. Gradual ramp with rollback criteria.
6. Full launch only after quality and guardrail review.

## 14. Monitoring checklist

### Data health

- Event volume and freshness.
- Schema changes.
- Missing/unknown rates.
- Label delay and label distribution.
- New-user/item rates.
- Feature-distribution drift.

### Training health

- Pipeline age and last successful run.
- Loss, NaN/Inf, gradients, throughput, and memory.
- Validation metrics by model/data version.
- Reproducibility and seed dispersion.

### Serving health

- Feature skew between training and inference.
- Prediction distribution and saturation.
- Candidate count and source coverage.
- p50/p95/p99 latency, errors, and timeouts.
- Model/index freshness.

### Product and safety health

- Engagement and satisfied-engagement metrics.
- Retention or return rate.
- Diversity, freshness, creator/item coverage.
- Hides, reports, complaints, and unsafe-content rates.
- Fairness or exposure metrics relevant to the product.

Monitor model, code, and data versions together. Google specifically recommends checking schema/feature skew, numerical stability, model age, performance, and live quality rather than waiting for obvious user-facing failure ([Monitoring ML pipelines](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)).

## 15. How the ML engineer works with autonomous agents

The agent proposes and accelerates work; deterministic software preserves scientific validity.

| Work | Agent responsibility | Deterministic ML responsibility |
| --- | --- | --- |
| Problem understanding | Summarize task and identify ambiguities | Validate benchmark contract |
| EDA | Interpret verified statistics | Read data and compute statistics |
| Research | Propose cited methods | Enforce allowed resources |
| Feature work | Specify or patch builders | Enforce time/leakage rules and materialize features |
| Training | Propose model/loss/config | Run isolated training and checkpointing |
| Evaluation | Interpret metrics and errors | Compute official metrics and alignment checks |
| Reflection | Accept/reject and select next hypothesis | Store immutable results and promote true best artifact |
| Recovery | Choose approved retry/repair | Enforce timeout, retry count, and rollback |
| Reporting | Explain outcomes | Derive tables from the registry |

Agents must not:

- Read hidden-test labels.
- Modify `evaluate.py` or split dates.
- Enter metric values manually.
- Use current-row outcomes as inference features.
- Promote a model that did not pass deterministic checks.
- Hide failed experiments or resource consumption.

## 16. Recommended experiment sequence for this repository

### Iteration 0: verify the harness

- Validate data files, schema, counts, and splits.
- Run random and popularity rungs.
- Reproduce the official FM validation score.
- Save a real FM checkpoint and generate a checked validation submission.

### Iteration 1: rank-aware tree model

- Build leakage-safe tabular features.
- Train LightGBM LambdaRank or XENDCG grouped by user.
- Tune ranking cutoff around nDCG@5.
- Compare GAUC, nDCG@5, segments, and runtime.

### Iteration 2: BPR-FM

- Reuse the FM representation.
- Sample positive/negative pairs within user.
- Compare negative-sampling strategies.
- Keep parameter count close to the baseline for a clean loss ablation.

### Iteration 3: chronological aggregate features

- User–author affinity.
- Recent duration/category preference.
- Smoothed item/author positive rate.
- Recency, trend, and cold-start indicators.

Add each family separately before combining them.

### Iteration 4: multi-task DeepFM

- Primary `long_view` head.
- Click and watch-time auxiliaries first.
- Add sparse deep-feedback tasks only if they help.
- Tune task weights and inspect negative transfer.

### Iteration 5: user-history attention

- Start with a bounded recent-history DIN-style encoder.
- Use only interactions before the prediction time.
- Compare against fixed aggregate histories at matched compute.

### Later iterations

- Censored watch-time auxiliary loss.
- MMoE if task interference is visible.
- Temporal weighting or rolling train windows.
- Rank-normalized ensemble of validation-diverse winners.
- Random-exposure diagnostics or OPE as an advanced analysis.

Do not precommit all 50 iterations. The next experiment should depend on the evidence from the preceding one.

## 17. Definition of done for an ML experiment

An experiment is complete only when:

- [ ] The hypothesis and success criterion were written before training.
- [ ] Data, code, configuration, environment, and seed are versioned.
- [ ] Leakage and alignment checks pass.
- [ ] Training finishes or the failure is classified and logged.
- [ ] Predictions have the expected row count and finite scores.
- [ ] Official validation metrics are recorded.
- [ ] Important segment metrics are reviewed.
- [ ] Runtime, tokens, GPU-hours, and peak memory are recorded.
- [ ] The candidate is compared with the incumbent and official baseline.
- [ ] The result is accepted, rejected, revised, or queued for ensemble.
- [ ] The next hypothesis follows from the evidence.

## 18. Definition of done for the final recommender

- [ ] Official baseline reproduced.
- [ ] Convergence or hard budget reached automatically.
- [ ] Validation-best—not last—checkpoint selected.
- [ ] Hidden test was never used for development feedback.
- [ ] Submission schema and alignment pass.
- [ ] Multi-seed stability is reported for the finalist when affordable.
- [ ] Full iteration and recovery log exists.
- [ ] Manual interventions and resource usage are reported.
- [ ] Setup and reproduction commands work from a clean environment.
- [ ] Limitations and likely next improvements are documented.

## 19. Primary references

- Google, [Recommendation systems overview](https://developers.google.com/machine-learning/recommendation/overview/types).
- Covington, Adams, and Sargin, [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).
- Google, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).
- Google, [Monitoring ML pipelines](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring).
- TensorFlow Recommenders, [Basic retrieval](https://www.tensorflow.org/recommenders/examples/basic_retrieval) and [multi-task recommenders](https://www.tensorflow.org/recommenders/examples/multitask/).
- Gao et al., [KuaiRand](https://arxiv.org/abs/2208.08696).
- Rendle et al., [Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618).
- Guo et al., [DeepFM](https://arxiv.org/abs/1703.04247).
- Zhou et al., [Deep Interest Network](https://arxiv.org/abs/1706.06978).
- Kang and McAuley, [SASRec](https://arxiv.org/abs/1808.09781).
- Ma et al., [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/).
- Ma et al., [ESMM](https://arxiv.org/abs/1804.07931).
- Zhao et al., [Counteracting Duration Bias via Censored Watch Time](https://arxiv.org/abs/2406.07932).
- LightGBM, [Ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
- Li et al., [Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Published-3.pdf).
