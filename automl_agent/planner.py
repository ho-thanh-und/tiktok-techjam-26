from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .config import AgentConfig, Experiment, placeholders
from .errors import ExecutionFailure
from .io_utils import atomic_write_json, read_json, utc_timestamp
from .runner import log_progress, render_command, run_command
from .storage import RunStore


@dataclass(frozen=True)
class PlanDecision:
    experiment: Experiment
    reason: str
    evidence: tuple[str, ...]
    planner_command_seconds: float = 0.0
    planner_llm_tokens: int = 0
    planner_mode: str = "catalog"


class CatalogResearchPlanner:
    """A deterministic, auditable research policy over an approved experiment catalog."""

    def choose(self, config: AgentConfig, state: dict[str, Any]) -> PlanDecision | None:
        completed = {item["experiment_id"] for item in state.get("experiments", [])}
        available = [item for item in config.experiments if item.experiment_id not in completed]
        if not available:
            return None

        family_outcomes: dict[str, list[dict[str, Any]]] = {}
        for record in state.get("experiments", []):
            family_outcomes.setdefault(record.get("family", "unknown"), []).append(record)

        def score(item: Experiment) -> tuple[int, int, str]:
            outcomes = family_outcomes.get(item.family, [])
            promoted = sum(1 for outcome in outcomes if outcome.get("status") == "promoted")
            failed = sum(
                1 for outcome in outcomes if outcome.get("status") in {"failed", "timed_out", "rejected"}
            )
            return (item.priority + promoted * 10 - failed * 4, -len(outcomes), item.experiment_id)

        selected = max(available, key=score)
        outcomes = family_outcomes.get(selected.family, [])
        evidence = [
            f"incumbent={state.get('incumbent', {}).get('experiment_id', 'official_baseline') if state.get('incumbent') else 'official_baseline'}",
            f"family_prior_runs={len(outcomes)}",
            f"remaining_catalog_candidates={len(available)}",
        ]
        if outcomes and outcomes[-1].get("status") == "promoted":
            reason = f"Exploit validated gains in the {selected.family} family"
        elif outcomes:
            reason = f"Revise the {selected.family} hypothesis after prior evidence"
        else:
            reason = f"Explore highest-priority untested family: {selected.family}"
        return PlanDecision(selected, reason, tuple(evidence))


class ExternalResearchPlanner:
    """Runs a sandboxed planner command and validates its choice against the catalog."""

    def __init__(self, config: AgentConfig, store: RunStore) -> None:
        self.config = config
        self.store = store
        self.fallback = CatalogResearchPlanner()

    def _evidence_pack(self, state: dict[str, Any]) -> dict[str, Any]:
        completed = {item["experiment_id"] for item in state.get("experiments", [])}
        candidates = [
            {
                "experiment_id": item.experiment_id,
                "family": item.family,
                "priority": item.priority,
                "hypothesis": item.hypothesis,
                "expected_effect": item.expected_effect,
                "knowledge_ids": list(item.knowledge_ids),
            }
            for item in self.config.experiments
            if item.experiment_id not in completed
        ]
        return {
            "schema_version": 1,
            "created_at": utc_timestamp(),
            "run_id": self.store.run_id,
            "benchmark": self.config.benchmark_name,
            "positive_label": self.config.positive_label,
            "metrics": list(self.config.metric_names),
            "selection_metric": self.config.selection_metric,
            "baseline": state.get("baseline"),
            "incumbent": state.get("incumbent"),
            "previous_experiments": state.get("experiments", []),
            "iterations_used": state.get("iterations_used", 0),
            "iteration_cap": self.config.iteration_cap,
            "candidates": candidates,
            "constraints": {
                "choose_exactly_one_candidate": True,
                "may_not_change_metrics_or_split": True,
                "hidden_test_available": False,
            },
        }

    def choose(self, config: AgentConfig, state: dict[str, Any]) -> PlanDecision | None:
        completed = {item["experiment_id"] for item in state.get("experiments", [])}
        available = {item.experiment_id: item for item in config.experiments if item.experiment_id not in completed}
        if not available:
            return None
        decision_number = int(state.get("iterations_used", 0)) + 1
        planner_dir = self.store.root / "planner" / f"decision-{decision_number:02d}"
        planner_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = planner_dir / "evidence.json"
        decision_path = planner_dir / "decision.json"
        atomic_write_json(evidence_path, self._evidence_pack(state))
        try:
            maximum_attempts = 2 if config.retry_transient_once else 1
            planner_started = time.monotonic()
            result = None
            successful_decision_path = None
            for attempt in range(1, maximum_attempts + 1):
                attempt_decision_path = planner_dir / f"decision-attempt-{attempt}.json"
                values = {
                    **placeholders(config),
                    "run_dir": str(self.store.root),
                    "evidence_path": str(evidence_path),
                    "decision_path": str(attempt_decision_path),
                }
                remaining_timeout = config.planner_timeout_seconds - (
                    time.monotonic() - planner_started
                )
                if remaining_timeout <= 0:
                    raise ExecutionFailure(
                        "Planner retry budget exhausted", failure_class="timeout"
                    )
                try:
                    result = run_command(
                        render_command(config.planner_command or (), values),
                        cwd=config.workspace,
                        output_dir=planner_dir / "execution" / f"attempt-{attempt}",
                        timeout_seconds=remaining_timeout,
                        poll_seconds=config.command_poll_seconds,
                        label=f"LLM planner decision {decision_number} (attempt {attempt}/{maximum_attempts})",
                    )
                    successful_decision_path = attempt_decision_path
                    break
                except ExecutionFailure as exc:
                    self.store.append_event(
                        "planner_attempt_failed",
                        {
                            "decision": decision_number,
                            "attempt": attempt,
                            "failure_class": exc.failure_class,
                            "error": str(exc),
                        },
                    )
                    if exc.failure_class != "transient" or attempt >= maximum_attempts:
                        raise
                    log_progress(
                        "RETRY",
                        "LLM planner transient failure",
                        decision=decision_number,
                        next_attempt=attempt + 1,
                        backoff_seconds=2,
                    )
                    time.sleep(min(2.0, max(0.0, remaining_timeout)))
            if result is None or successful_decision_path is None:
                raise ExecutionFailure("Planner produced no successful attempt")
            if not successful_decision_path.is_file():
                raise ExecutionFailure(
                    "Planner command did not create decision JSON", failure_class="schema_alignment"
                )
            raw = read_json(successful_decision_path)
            atomic_write_json(decision_path, raw)
            experiment_id = raw.get("experiment_id")
            if experiment_id not in available:
                raise ExecutionFailure(
                    f"Planner selected unknown, completed, or disallowed experiment: {experiment_id!r}",
                    failure_class="policy",
                )
            reason = raw.get("reason")
            evidence = raw.get("evidence", [])
            resources = raw.get("resources", {})
            if not isinstance(reason, str) or not reason.strip():
                raise ExecutionFailure("Planner reason must be a non-empty string", failure_class="schema_alignment")
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                raise ExecutionFailure("Planner evidence must be a string array", failure_class="schema_alignment")
            if not isinstance(resources, dict):
                raise ExecutionFailure("Planner resources must be an object", failure_class="schema_alignment")
            llm_tokens = resources.get("llm_tokens", 0)
            if not isinstance(llm_tokens, int) or llm_tokens < 0:
                raise ExecutionFailure(
                    "Planner llm_tokens must be a non-negative integer", failure_class="schema_alignment"
                )
            return PlanDecision(
                experiment=available[experiment_id],
                reason=reason,
                evidence=tuple(evidence),
                planner_command_seconds=time.monotonic() - planner_started,
                planner_llm_tokens=llm_tokens,
                planner_mode=config.planner_mode,
            )
        except ExecutionFailure as exc:
            self.store.append_event(
                "planner_command_failed",
                {"error": str(exc), "failure_class": exc.failure_class},
            )
            if not config.planner_fallback_to_catalog:
                raise
            fallback = self.fallback.choose(config, state)
            if fallback is None:
                return None
            return PlanDecision(
                experiment=fallback.experiment,
                reason=f"External planner failed safely; {fallback.reason}",
                evidence=(*fallback.evidence, f"planner_failure_class={exc.failure_class}"),
                planner_mode="catalog_fallback",
            )
