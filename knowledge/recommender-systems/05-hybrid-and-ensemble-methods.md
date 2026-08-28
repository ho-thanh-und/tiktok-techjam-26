---
knowledge_id: rs_hybrid_ensemble
tags: [hybrid, ensemble, blending, stacking, switching]
agent_use: [D3, D5, D12]
relevance: high
---

# Hybrid and Ensemble Methods

## Combination patterns

| Pattern | Use | Important requirement |
|---|---|---|
| Weighted blend | Combine several scores | Normalize/calibrate score scales; fit weights on held-out data |
| Switching | Route examples/segments to different models | Reliable routing feature and fallback |
| Cascade | A broad model passes candidates to a refinement model | Earlier stage must retain useful candidates |
| Feature augmentation | Add one model's output as another model's feature | Out-of-fold predictions to prevent leakage |
| Meta-level | Feed learned representations into another learner | Stable representation and separate validation |
| Feature combination | Train one model over collaborative and content features | Model must express meaningful interactions |
| Mixed presentation | Surface outputs from multiple recommenders | Mainly a product/serving choice |

## Agent decision rule

Ensemble only when component errors are complementary. Two models with almost identical ranking errors rarely justify added complexity. Compare per-user score deltas and segment behavior, not just aggregate scores.

## KuaiRand candidates

- Blend a pairwise factor/FM model that improves GAUC with a listwise/tree ranker that improves nDCG@5.
- Switch to content/context scoring for cold IDs while using collaborative scoring for warm IDs.
- Stack out-of-fold collaborative scores into a feature-rich ranker.
- Use a cascade only if a future system adds retrieval; the current challenge already supplies candidates.

## D5 search space

- normalized rank, z-score, logit, or calibrated-probability inputs;
- non-negative blend weights constrained to sum to one;
- segment-specific weights only when validation support is sufficient;
- cold-start routing thresholds;
- simple linear blend before a learned meta-model.

## Guardrails

- Fit weights and thresholds without touching hidden test outcomes.
- Produce stacking features out of fold.
- Report each component, the blend, correlation, latency, and complexity.
- Reject a blend if its gain is within uncertainty or relies on a fragile segment.
