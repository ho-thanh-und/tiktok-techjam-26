---
knowledge_id: rs_content_features
tags: [content-based, features, side-information, cold-start]
agent_use: [D3, D4, D5]
relevance: high
---

# Content and Feature Engineering

## Content pipeline

A content-aware recommender typically has three stages:

1. extract and preprocess domain-specific item/user attributes;
2. learn user profiles or affinity representations offline;
3. score candidates efficiently at inference.

Feature values and fields differ in importance. Selection, transformation, and weighting should be justified by the recommendation task, not by availability alone.

## KuaiRand translation

Static user, video, and author fields are side information. A feature that is constant across all rows of one user cannot change that user's ranking unless it interacts with item or context. Useful candidates therefore include:

- user × video/category/author affinities;
- past user engagement aggregated by item attribute;
- candidate attribute × context crosses;
- cold-start indicators and frequency/support features;
- learned embeddings of categorical side information;
- candidate similarity to a past-only user profile.

Blindly adding raw side columns has already shown limited value in the starter kit. This does not falsify content information; it suggests that representation, interaction, support, or validation may be the limiting factor.

## Feature acceptance checklist

- Available at the prediction timestamp?
- Computed only from earlier events?
- Varies within the ranked user group, or interacts with something that does?
- Has sufficient support and stable missing-value semantics?
- Adds value beyond ID/frequency baselines?
- Helps the intended segment without unacceptable aggregate damage?
- Reproducible in training, validation, test, and serving?

## Experiment patterns

- Add one coherent feature family at a time.
- Compare raw field, embedding, affinity, and crossed variants.
- Report cold/warm and head/tail effects.
- Use feature ablations after a group succeeds.
- For a failed feature, distinguish “no information” from “wrong representation.”

## Model-selection note

Classical text-processing features can provide cheap baselines, while modern encoders may capture richer semantics at greater cost. Select either approach using current evidence, a controlled comparison, and the domain-specific need to weight or select attributes.
