---
knowledge_id: rs_evaluation_bias
tags: [evaluation, ranking-metrics, temporal-split, selection-bias, leakage]
agent_use: [D1, D5, D9, D10]
relevance: critical
---

# Evaluation and Bias

## Evaluation contract

Recommenders can be assessed through user studies, online experiments, or offline historical logs. This repository uses offline evaluation, so conclusions are about performance on logged candidate exposures—not necessarily catalogue-wide or online utility.

Recommendation quality has dimensions beyond predictive error: coverage, confidence/trust, novelty, serendipity, diversity, robustness, stability, and scalability. The benchmark promotion rule remains fixed, but these dimensions can be recorded as diagnostic constraints.

## Metric alignment

- GAUC asks whether positives outrank negatives within each eligible user and aggregates across users.
- nDCG@5 emphasizes the ordering near the visible top of each user's list.
- AUC-like measures consider all pair inversions more evenly; top-heavy metrics care more about early positions.
- Pointwise log loss can be useful for calibration yet misaligned with rank order.

Always report GAUC, nDCG@5, and their fixed mean separately. A gain in the mean can hide a meaningful tradeoff.

## Split and leakage rules

- Measure on examples excluded from fitting and tuning.
- Prefer temporal validation when deployment predicts later behavior from earlier history.
- Build aggregates and sequences inside each training fold, then apply them forward.
- Same-row outcomes and future interactions are never features.
- Keep the official evaluator unchanged; validate row alignment and user grouping deterministically.

## Selection and exposure bias

Observed interactions are missing-not-at-random: users only react to shown content, and historical recommenders affect what is shown. Popular items and active users can be overrepresented. Random holdout may reproduce historical-policy bias and overstate generalization.

KuaiRand's randomized exposures are useful for bias diagnostics and off-policy research. However, propensity adjustment or counterfactual evaluation requires explicit assumptions and should not silently replace the official benchmark score.

## Required segment report

- mixed-label, all-positive, and all-negative users;
- low/medium/high user activity;
- head/torso/tail item popularity;
- cold versus warm users/items;
- date, time-of-day, tab, and major context slices;
- short versus long history;
- prediction coverage and score distribution.

## D10 interpretation questions

1. Did the intended metric and segment improve?
2. Is the change larger than seed/fold uncertainty?
3. Did coverage, calibration, or another metric degrade?
4. Could exposure, popularity, time, or leakage explain the gain?
5. Which hypothesis was supported or falsified?
