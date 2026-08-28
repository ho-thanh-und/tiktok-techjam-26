---
knowledge_id: rs_neighborhood_cf
tags: [collaborative-filtering, neighborhoods, SLIM, baselines]
agent_use: [D3, D5]
relevance: medium
---

# Neighborhood Collaborative Filtering

## Core idea

User-based methods score from similar users; item-based methods score from items related to a candidate. They are intuitive and locally explainable, but similarity becomes unreliable when user–item overlap is sparse. Large item catalogues also create coverage, scale, and popularity-bias problems.

Similarity should be paired with support awareness. A high similarity estimated from very few co-interactions is not equivalent to the same value estimated from many observations. Regularization or significance weighting can reduce this instability.

SLIM replaces a fixed similarity formula with learned sparse item–item linear coefficients. It can be used as a standalone scorer or a feature/model component in an ensemble.

## When to test

- A cheap interpretable collaborative baseline is missing.
- Item co-interaction density is high enough to estimate useful relations.
- A latent model lacks local co-occurrence signal.
- A diversity-aware ensemble needs a structurally different model.

## Tuning decisions

- user-based versus item-based neighborhood;
- similarity function and normalization;
- minimum support/significance shrinkage;
- neighborhood size;
- recency weighting;
- SLIM sparsity and coefficient regularization;
- whether scores become direct predictions, rank features, or blend inputs.

## Risks and checks

- Measure coverage: missing predictions can silently favor popular items.
- Evaluate head, torso, and tail items separately.
- Compute all interactions from training history only.
- Do not assume explainability implies better ranking metrics.
- Retain only if it adds standalone gain or complementary ensemble errors.
