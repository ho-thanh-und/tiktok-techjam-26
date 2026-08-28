---
knowledge_id: rs_problem_feedback
tags: [problem-framing, feedback, implicit-feedback, cold-start]
agent_use: [D0, D3, D4]
relevance: high
---

# Problem Framing and Feedback

## Durable principles

Recommender inputs fall into three broad families:

- collaborative signals from user–item interactions;
- content signals from user/item attributes;
- explicit requirements or domain knowledge.

Hybrid methods combine these families when their failure modes differ. Collaborative models capture community preference but struggle when overlap is sparse. Content models support new items and explanations but can over-specialize around known attributes.

Feedback can be explicit (ratings) or implicit (views, clicks, purchases, watch behavior). With unary implicit feedback, an observed action is evidence of interest or engagement, but a missing action is ambiguous. It may mean dislike, no exposure, or no opportunity. Therefore, missing rows should not automatically receive the same certainty as observed negatives.

## KuaiRand application

The current task is an impression-ranking problem: candidate rows already exist and `long_view` is the fixed binary target. This differs from full-catalog retrieval. The model needs to order each user's supplied impressions, not generate a catalogue-wide candidate set.

The agent should classify each signal before using it:

| Signal | Role | Caution |
|---|---|---|
| `long_view` | Primary supervised target | Must remain the scored objective |
| Other behavior labels | Auxiliary target or historical feature | Same-row outcomes would leak at inference |
| User/video/author attributes | Content/side information | Often need interactions to affect within-user rank |
| Logged impressions | Exposure-conditioned candidates | Reflect the historical logging policy |
| Past sequences | Collaborative/context evidence | Must be cut off before the prediction event |

## Decision implications

- If cold-start segments are weak, route to content or switching hybrids rather than forcing an ID-only model.
- If collaborative overlap is adequate, compare neighborhood and latent-factor signals.
- If implicit histories are used, specify the meaning and weight of zero/missing entries.
- Treat time, context, cold start, robustness, and multi-feedback as distinct experiment dimensions rather than one generic feature bucket.

## Testable hypotheses

- A past-only weighted history representation improves mixed-label users because it adds confidence-weighted collaborative evidence.
- Side information helps cold items more than head items; test segment interaction rather than only aggregate gain.
- Auxiliary labels help through representation sharing but should not replace `long_view` in the promotion rule.
