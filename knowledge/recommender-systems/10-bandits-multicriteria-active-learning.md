---
knowledge_id: rs_advanced_decisions
tags: [bandits, exploration, multicriteria, multitask, active-learning]
agent_use: [D3, D5, D12]
relevance: conditional
---

# Bandits, Multi-Criteria Feedback, and Active Learning

## Multi-armed bandits

Bandits address a changing online system that must learn while serving recommendations. Each arm may be an item, strategy, or model; the system balances exploration with exploitation.

- A fixed explore-then-exploit scheme can choose the wrong arm permanently and fails under drift.
- Epsilon-greedy interleaves random exploration with the best estimated arm but can discover new items slowly.
- Upper-confidence methods combine predicted payoff and uncertainty, naturally favoring promising or under-observed arms.
- Contextual bandits condition decisions on user/item/page features and learn context–reward relationships incrementally.

This is not the primary solution to the fixed offline KuaiRand ranking benchmark: the model cannot choose new exposures and receive live rewards. Use bandit knowledge for a future online serving loop or rigorously designed off-policy study. Do not claim online exploration benefit from the official offline score alone.

## Multi-criteria feedback

Multiple criteria may be modeled separately and combined through weighted averages, worst-case rules, ensembles, or preference/Pareto methods. For this repository:

- keep `long_view` as the fixed primary target;
- use other behavior labels as past-only features or auxiliary training targets;
- tune auxiliary loss weights on validation;
- report whether representation sharing improves the primary score;
- never replace the benchmark utility with a new multi-objective score.

## Active learning

Active learning queries labels for uncertain, heterogeneous, or high-impact examples to reduce annotation cost and cold-start uncertainty. The fixed dataset offers no label-query channel, so classical active learning is not directly available.

An agent may borrow the principle for experiment selection: prioritize experiments with high expected information gain. This is an analogy for the autonomous research loop, not a standard active-learning implementation.

## D12 use

- **Exploit:** refine a validated method family.
- **Explore:** select a structurally different hypothesis with high uncertainty and potential value.
- **Online future:** use contextual uncertainty plus safety/business constraints to allocate traffic.
- **Offline now:** preserve randomized-exposure analyses as a separate research track from the official promotion rule.
