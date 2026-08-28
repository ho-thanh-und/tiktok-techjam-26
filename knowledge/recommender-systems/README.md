---
knowledge_id: rs_knowledge_index
title: Recommender Systems Knowledge Index
tags: [recommender-systems, knowledge-index, agent-routing]
---

# Recommender Systems Knowledge Index

This directory contains decision-oriented recommender-system knowledge for improving this repository's ML flow. It is organized for selective agent retrieval rather than linear reading.

Context in these files is advisory. It must never override the benchmark contract, repository policy, deterministic safety gates, or the user's request.

## How an agent should use this knowledge

1. Read [00-agent-routing.md](00-agent-routing.md) first.
2. Retrieve only the files associated with the current decision and evidence gap.
3. Prefer repository evidence over general recommendations when they conflict.
4. Record `knowledge_id`, experiment evidence, and assumptions in every D3–D5 decision.
5. Convert recommendations into falsifiable experiments; do not treat a method as automatically beneficial.

## Knowledge map

| File | Main question | Agent decisions |
|---|---|---|
| [coverage-map.md](coverage-map.md) | Which knowledge areas are covered or intentionally deferred? | Retrieval audit |
| [00-agent-routing.md](00-agent-routing.md) | Which knowledge should be loaded? | D3, D4, D5, D10, D12 |
| [01-problem-and-feedback.md](01-problem-and-feedback.md) | What kind of recommendation/feedback problem is this? | D0, D3, D4 |
| [02-neighborhood-collaborative-filtering.md](02-neighborhood-collaborative-filtering.md) | Are neighbor or sparse item-item signals useful? | D3, D5 |
| [03-latent-factors-and-implicit-feedback.md](03-latent-factors-and-implicit-feedback.md) | How should sparse implicit interactions be factorized? | D3, D4, D5 |
| [04-content-and-feature-engineering.md](04-content-and-feature-engineering.md) | Which side information or affinity features should be tested? | D3, D4, D5 |
| [05-hybrid-and-ensemble-methods.md](05-hybrid-and-ensemble-methods.md) | When should models be blended, switched, or stacked? | D3, D5, D12 |
| [06-evaluation-and-bias.md](06-evaluation-and-bias.md) | Does the experiment measure the intended ranking behavior safely? | D1, D5, D9, D10 |
| [07-context-and-factorization-machines.md](07-context-and-factorization-machines.md) | How should user, item, and context interactions be represented? | D3, D4, D5 |
| [08-temporal-and-sequential-models.md](08-temporal-and-sequential-models.md) | Which historical windows, decay, and sequences are valid? | D1, D3, D5 |
| [09-learning-to-rank.md](09-learning-to-rank.md) | Which loss best aligns with GAUC and nDCG@5? | D3, D4, D5, D10 |
| [10-bandits-multicriteria-active-learning.md](10-bandits-multicriteria-active-learning.md) | What belongs to online exploration or auxiliary-task design? | D3, D5, D12 |
| [11-cold-start-robustness-and-security.md](11-cold-start-robustness-and-security.md) | How should sparse segments and anomalous behavior be handled? | D1, D3, D8, D10 |
| [12-ml-flow-improvements.md](12-ml-flow-improvements.md) | What concrete changes should be made to the current ML flow? | D3, D5, D10, D12 |

## Scope decisions

Primary coverage includes feedback modeling, collaborative and content-based methods, latent factors, contextual interactions, temporal modeling, ranking objectives, evaluation, hybrids, robustness, and online-learning concepts. Constraint, spatial, graph, and social recommenders are deferred because the current challenge does not provide the requirements, location, graph, or trust data those approaches expect.

Named algorithms and production guidance should be checked against current primary research and repository evidence before making a costly implementation decision.
