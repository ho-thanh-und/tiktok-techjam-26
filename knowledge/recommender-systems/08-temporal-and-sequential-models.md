---
knowledge_id: rs_temporal_sequence
tags: [temporal, sequential, recency, drift, markov]
agent_use: [D1, D3, D5]
relevance: high
---

# Temporal and Sequential Models

## What changes over time

User preferences drift, item popularity changes, and context can be periodic. Long histories are stable but may be stale; short histories are responsive but noisy. Time should therefore be treated as a modeling and validation dimension.

## Method families

- recency weighting or decay;
- fixed and multiple historical windows;
- periodic context such as hour, day, or season;
- time-dependent biases/factors, including Time-SVD++-style ideas;
- first-order or higher-order Markov transitions;
- selective/variable history length to control sparsity;
- sequential pattern features.

## Leakage-safe KuaiRand flow

For an event at time `t`:

1. sort history deterministically;
2. retain only events strictly earlier than `t` under the repository's tie rule;
3. build user/item/author/category counts and rates at several windows;
4. compute time since last event and recent attribute affinities;
5. score the current candidate;
6. update state only after prediction when simulating online inference.

Use temporal validation to choose windows and decay. Random splitting can leak later preferences into earlier examples even when the target column itself is absent.

## D5 search space

- window lengths: short, medium, and all-history;
- exponential decay half-life;
- recent sequence length;
- attribute-level versus exact-item transitions;
- first-order versus selective higher-order histories;
- separate positive-action and exposure histories;
- pooling: count, rate, recency, mean embedding, attention, or transition score.

## Decision signals

Test temporal features when performance varies by date/time, recent users outperform stale users, or item popularity drifts. Prefer simple past-only aggregates before a large sequence model. Promote a sequential model only if its incremental gain survives time slices and compute/latency constraints.
