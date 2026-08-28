---
knowledge_id: rs_context_fm
tags: [context-aware, factorization-machines, interactions, sparse-features]
agent_use: [D3, D4, D5]
relevance: critical
---

# Context and Factorization Machines

## Context-aware recommendation

The scoring function can be viewed as user × item × context → score. Context may be added by:

- pre-filtering history to a matching context;
- post-filtering or re-ranking a base model's output;
- modeling context directly with user and item features.

Overly narrow context slices create sparse, high-variance estimates. Use hierarchical fallbacks or shrink fine-grained estimates toward broader aggregates.

## Factorization machines

An FM represents a row as a sparse feature vector and models:

- a global intercept;
- first-order feature effects;
- low-rank pairwise feature interactions.

Rather than fitting a separate coefficient for every possible pair, each feature has a latent vector whose dot product defines its interaction. This yields roughly O(k × nonzero features) scoring and works well for sparse user, item, and context identifiers.

FMs accept binary, set-valued, and real-valued features. They can express matrix-factorization-like interactions and incorporate side information in the same model. Logistic and pairwise ranking objectives are both possible.

## KuaiRand interaction map

High-value interaction blocks to test include:

- user ID × video ID;
- user ID × author/category/tag;
- user profile × video/category/author;
- past user profile/affinity × candidate attributes;
- video/author × tab/time/context;
- user/history state × context;
- candidate popularity/support × user activity or context.

First-order user-only features are constant within a user's candidate list and cannot affect within-user ranking. Their value must come through interactions, calibration, or cross-user optimization.

## D5 tuning knobs

- factor dimension;
- separate regularization for linear and interaction terms;
- enabled interaction blocks;
- field-aware or higher-order variants only after pairwise FM is strong;
- negative/pair sampling and loss;
- frequency thresholds and rare-category hashing/bucketing;
- score calibration when blending.

## Risks

- Allowing every field pair to interact can add noise and overfit.
- Higher-order interactions increase cost and are not automatically useful.
- Feature engineering determines what an FM can learn; the model does not repair leaked or semantically wrong inputs.
