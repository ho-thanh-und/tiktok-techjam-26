---
knowledge_id: rs_learning_to_rank
tags: [learning-to-rank, pairwise, listwise, BPR, GAUC, NDCG]
agent_use: [D3, D4, D5, D10]
relevance: critical
---

# Learning to Rank

## Why objective alignment matters

Users see ordered lists, not raw predicted values. Minimizing pointwise prediction error does not directly optimize which candidates appear first. Ranking metrics are non-smooth, so practical systems use pointwise, pairwise, or listwise surrogate objectives.

## Objective families

| Family | Training unit | Strength | Main risk |
|---|---|---|---|
| Pointwise | One impression | Simple, scalable, calibratable | Ignores within-user competition |
| Pairwise | Positive/negative pair for one user | Directly reduces inversions; aligns with AUC/GAUC | Pair volume and sampling bias |
| Listwise | Candidate list for one user | Can emphasize top-list quality/NDCG | Variable groups, cost, unstable small lists |

Pairwise rank learning constructs comparisons between items for the same user. BPR is a natural implicit-feedback objective. Listwise methods optimize a surrogate over the full group or weight pair changes by their ranking impact.

## Recommended D3 branch for this repository

The current pointwise FM baseline and rank-based evaluation create a clear hypothesis: use the same features/model capacity while changing only the objective and grouping.

Suggested controlled ladder:

1. pointwise incumbent;
2. within-user BPR with uniform valid negatives;
3. BPR with tuned/semi-hard negatives;
4. listwise softmax or LambdaRank-style objective aligned near cutoff 5;
5. blend pairwise and top-heavy models if error tradeoffs are complementary.

## Pair construction guardrails

- Pair only candidates belonging to the same user group.
- Build pairs inside the training fold.
- Define all-positive/all-negative group behavior explicitly; they contain no binary ordering pairs.
- Cap or sample pairs so very large groups do not dominate.
- Record sampling distribution and seed.
- Avoid future/same-row auxiliary outcomes when selecting hard negatives.

## Evaluation interpretation

- Pairwise gains should first appear in GAUC for mixed-label users.
- Listwise/top-heavy gains should first appear in nDCG@5.
- A GAUC gain with nDCG loss is partial support, not automatic failure; test top-weighting or blending.
- A train-only ranking gain suggests sampling, regularization, grouping, or leakage issues.
