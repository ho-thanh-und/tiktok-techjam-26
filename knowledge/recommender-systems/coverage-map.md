---
knowledge_id: rs_knowledge_coverage
tags: [knowledge-coverage, scope, retrieval]
agent_use: [retrieval-audit]
relevance: reference
---

# Knowledge Coverage Map

## Purpose

This file records which recommender-system knowledge is available to the agents and which topics are intentionally deferred. It helps retrieval avoid loading irrelevant context.

## Coverage decisions

| Knowledge area | Relevance to current flow | Knowledge destination |
|---|---|---|
| Problem framing and implicit feedback | Defines target and missing-signal semantics | `01` |
| Neighborhood collaborative filtering | Collaborative baselines, similarity, SLIM | `02` |
| Latent-factor models | Factors, regularization, implicit history | `03` |
| Content-based recommendation | Side information and feature workflow | `04` |
| Hybrid and ensemble methods | Blending, switching, stacking | `05` |
| Evaluation and bias | Metrics, temporal testing, selection bias | `06` |
| Context-aware modeling | Context and factorization machines | `07` |
| Temporal and sequential methods | Drift, history windows, transitions | `08` |
| Learning to rank | Pointwise, pairwise, and listwise objectives | `09` |
| Online and multi-feedback decisions | Bandits, auxiliary criteria, active learning | `10` |
| Cold start and robustness | Fallbacks, anomaly and stability checks | `11` |
| Constraint-based recommendation | Requires explicit user requirements | Deferred |
| Spatial recommendation | Requires meaningful location inputs | Deferred |
| Graph recommendation | Requires a supported graph use case | Deferred |
| Social and trust recommendation | Requires social or trust data | Deferred |

Numbers such as `01` refer to the numbered Markdown filenames in this directory.

## Inclusion standard

A section was promoted when it could change at least one of:

- task/feedback interpretation;
- feature construction;
- objective or sampling;
- model and interaction design;
- temporal validity;
- evaluation and result interpretation;
- cold-start/robustness handling;
- experiment portfolio decisions.

Deferred areas can be added if the system gains requirements, graph, social, trust, spatial, group, privacy, or reciprocal-recommendation inputs.
