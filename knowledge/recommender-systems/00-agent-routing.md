---
knowledge_id: rs_agent_routing
tags: [agent-routing, retrieval, D3, D5]
agent_use: [D3, D4, D5, D10, D12]
relevance: critical
---

# Agent Routing Guide

## Retrieval rule

Use structured retrieval over these Markdown files before considering a broad semantic search over all available context. The agent normally needs one to three compact decision files, the repository's task contract, and the latest experiment evidence. Broader search is a fallback for questions not covered here.

## Route by observed evidence

| Evidence or question | Load | Likely action family |
|---|---|---|
| GAUC weak; pointwise loss is current baseline | `09`, then `06` | Pairwise within-user or listwise training |
| nDCG@5 weak or GAUC improves while nDCG falls | `09`, `06` | Top-heavy/listwise objective, pair sampling, blend |
| Sparse IDs or weak collaborative coverage | `03`, then `04` | Regularized factors plus side information |
| Side features add no gain | `04`, `07`, `06` | Test interactions/affinities; check leakage and split |
| User/item/context crosses are missing | `07` | FM/DeepFM interaction design |
| Drift, time-of-day, or history-length segment gap | `08`, then `06` | Past-only windows, decay, sequential features |
| Cold users/items underperform | `04`, `05`, `11` | Content fallback or switching hybrid |
| Two strong models have complementary errors | `05`, `06` | Calibrated blend, stack, or conditional switch |
| Unexpected offline gain from exposure/popularity | `06` | Selection-bias audit and temporal re-test |
| Suspicious bursts, duplicates, or extreme profiles | `11` | Data-quality/anomaly diagnosis; no automatic deletion |
| Online exploration/new-item problem | `10` | Contextual bandit design—not current offline scoring |
| Auxiliary feedback signals are available | `10`, then `09` | Multi-task/criteria features with `long_view` primary |

File numbers above refer to filenames in this directory.

## Minimum evidence pack before D3

The research planner should receive:

- fixed task contract and metric implementation;
- incumbent and baseline scores for GAUC, nDCG@5, and their mean;
- scores by user activity, item popularity, label mix, and time/context segments;
- data sizes, sparsity, cold-start rates, and feature availability at inference;
- current objective, candidate grouping, negative sampling, and model family;
- previous experiments, including failed ideas and uncertainty across seeds/folds;
- compute/time budget and permitted file/code scope.

## D3–D5 output contract

Every proposal should contain:

```yaml
decision:
  decision_id: unique_id
  stage: D3|D4|D5
  evidence_used: []
  knowledge_ids: []
  choice: concise_action
  hypothesis: falsifiable_reason
  primary_metric_expected: GAUC|nDCG@5|primary
  risk: []
  experiment:
    control: incumbent_config
    changed_variables: []
    fixed_variables: []
    search_space: {}
    success_rule: numeric_and_segment_rule
    falsification_rule: numeric_rule
  assumptions: []
```

## Guardrails

- Never infer dislike from an unobserved interaction without defining confidence/weighting.
- Never use future or same-row outcome information to build features.
- Never change the fixed target, user-grouping rule, metric code, or official split merely because a source suggests a different setup.
- Do not choose by one aggregate metric alone; inspect GAUC and nDCG@5 separately.
- Do not retrieve unrelated topics as filler. Missing evidence should be reported as missing.
- Do not convert a source statement into a repository fact unless it was verified on this dataset.
