from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def choose(evidence: dict[str, Any]) -> dict[str, Any]:
    candidates = evidence.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("EvidencePack contains no permitted candidates")
    outcomes: dict[str, list[dict[str, Any]]] = {}
    for previous in evidence.get("previous_experiments", []):
        outcomes.setdefault(str(previous.get("family", "unknown")), []).append(previous)

    def candidate_score(candidate: dict[str, Any]) -> tuple[int, int, str]:
        family = str(candidate["family"])
        prior = outcomes.get(family, [])
        promoted = sum(item.get("status") == "promoted" for item in prior)
        failed = sum(
            item.get("status") in {"failed", "timed_out", "rejected"} for item in prior
        )
        return (
            int(candidate["priority"]) + promoted * 10 - failed * 4,
            -len(prior),
            str(candidate["experiment_id"]),
        )

    selected = max(candidates, key=candidate_score)
    family = str(selected["family"])
    prior = outcomes.get(family, [])
    if prior and prior[-1].get("status") == "promoted":
        action = "exploit a validated family"
    elif prior:
        action = "revise a family after negative evidence"
    else:
        action = "explore the highest-priority untested family"
    return {
        "schema_version": 1,
        "experiment_id": selected["experiment_id"],
        "reason": f"Choose {selected['experiment_id']} to {action}",
        "evidence": [
            f"incumbent={evidence.get('incumbent', {}).get('experiment_id', 'official_baseline')}",
            f"family_prior_runs={len(prior)}",
            f"available_candidates={len(candidates)}",
        ],
        "resources": {"llm_tokens": 0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditable experiment-catalog research planner")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args(argv)
    with Path(args.evidence).open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    decision = choose(evidence)
    output = Path(args.decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

