from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

from .budget import converged, require_capacity, snapshot, update_convergence
from .config import AgentConfig, Experiment, placeholders
from .contracts import validate_contract
from .errors import AgentError, BudgetError, ContractError, ExecutionFailure
from .io_utils import atomic_write_json, read_json, sha256_file, utc_timestamp
from .planner import CatalogResearchPlanner, ExternalResearchPlanner, PlanDecision
from .runner import render_command, run_command
from .storage import RunStore


TERMINAL_RUN_STATUSES = {"completed", "blocked_contract", "failed"}


def _score_from_result(
    config: AgentConfig, result: dict[str, Any]
) -> tuple[dict[str, float], float, dict[str, float | int]]:
    if result.get("status") != "succeeded":
        raise ExecutionFailure("Result JSON status must be 'succeeded'", failure_class="schema_alignment")
    raw_metrics = result.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ExecutionFailure("Result JSON metrics must be an object", failure_class="schema_alignment")
    metrics: dict[str, float] = {}
    for name in config.metric_names:
        value = raw_metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExecutionFailure(f"Missing or invalid metric {name}", failure_class="schema_alignment")
        metrics[name] = float(value)
    if config.selection_metric == "official_selection_score":
        value = result.get("official_selection_score")
    else:
        value = metrics.get(config.selection_metric)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExecutionFailure(
            f"Missing or invalid selection metric {config.selection_metric}",
            failure_class="schema_alignment",
        )
    raw_resources = result.get("resources", {})
    if not isinstance(raw_resources, dict):
        raise ExecutionFailure("Result resources must be an object", failure_class="schema_alignment")
    gpu_hours = raw_resources.get("gpu_hours", 0.0)
    llm_tokens = raw_resources.get("llm_tokens", 0)
    if not isinstance(gpu_hours, (int, float)) or not math.isfinite(float(gpu_hours)) or gpu_hours < 0:
        raise ExecutionFailure("resources.gpu_hours must be finite and non-negative", failure_class="schema_alignment")
    if not isinstance(llm_tokens, int) or llm_tokens < 0:
        raise ExecutionFailure("resources.llm_tokens must be a non-negative integer", failure_class="schema_alignment")
    return metrics, float(value), {"gpu_hours": float(gpu_hours), "llm_tokens": llm_tokens}


def _better(config: AgentConfig, candidate: float, incumbent: float) -> bool:
    return candidate > incumbent if config.maximize else candidate < incumbent


class AutonomousRun:
    def __init__(
        self,
        config: AgentConfig,
        store: RunStore,
        *,
        planner: CatalogResearchPlanner | ExternalResearchPlanner | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.planner = planner or (
            ExternalResearchPlanner(config, store)
            if config.planner_mode in {"command", "llm"}
            else CatalogResearchPlanner()
        )
        self.contract_report: dict[str, Any] | None = None

    def _render_values(self, **extra: str) -> dict[str, str]:
        return {**placeholders(self.config), "run_dir": str(self.store.root), **extra}

    def _execute_with_recovery(
        self,
        command: tuple[str, ...],
        *,
        output_dir: Path,
        timeout_seconds: int,
        values: dict[str, str],
    ) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        maximum_attempts = 2 if self.config.retry_transient_once else 1
        total_started = time.monotonic()
        for attempt in range(1, maximum_attempts + 1):
            attempt_dir = output_dir / f"attempt-{attempt}"
            argv = render_command(command, values)
            started = time.monotonic()
            remaining_timeout = timeout_seconds - (started - total_started)
            if remaining_timeout <= 0:
                exc = ExecutionFailure(
                    f"Recovery budget exhausted after {started - total_started:.1f}s",
                    failure_class="timeout",
                )
                exc.attempts = attempts  # type: ignore[attr-defined]
                raise exc
            try:
                command_result = run_command(
                    argv,
                    cwd=self.config.workspace,
                    output_dir=attempt_dir,
                    timeout_seconds=remaining_timeout,
                    poll_seconds=self.config.command_poll_seconds,
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "succeeded",
                        "argv": list(argv),
                        "elapsed_seconds": command_result.elapsed_seconds,
                        "stdout": str(command_result.stdout_path),
                        "stderr": str(command_result.stderr_path),
                    }
                )
                return attempts
            except ExecutionFailure as exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "failure_class": exc.failure_class,
                        "error": str(exc),
                        "argv": list(argv),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                if exc.failure_class != "transient" or attempt >= maximum_attempts:
                    exc.attempts = attempts  # type: ignore[attr-defined]
                    raise
                self.store.append_event(
                    "transient_retry",
                    {"attempt": attempt, "command": list(argv), "error": str(exc)},
                )
        raise AssertionError("unreachable")

    @staticmethod
    def _account_resources(
        state: dict[str, Any],
        *,
        resources: dict[str, float | int],
        attempts: list[dict[str, Any]],
        validation_attempts: list[dict[str, Any]],
    ) -> None:
        totals = state.setdefault(
            "resources", {"command_seconds": 0.0, "gpu_hours": 0.0, "llm_tokens": 0}
        )
        totals["command_seconds"] = float(totals.get("command_seconds", 0.0)) + sum(
            float(item.get("elapsed_seconds", 0.0)) for item in attempts + validation_attempts
        )
        totals["gpu_hours"] = float(totals.get("gpu_hours", 0.0)) + float(
            resources.get("gpu_hours", 0.0)
        )
        totals["llm_tokens"] = int(totals.get("llm_tokens", 0)) + int(
            resources.get("llm_tokens", 0)
        )

    def _validate_submission(self, submission_path: Path, output_dir: Path) -> list[dict[str, Any]]:
        if not submission_path.is_file():
            raise ExecutionFailure(
                f"Expected submission was not created: {submission_path}",
                failure_class="schema_alignment",
            )
        values = self._render_values(submission_path=str(submission_path))
        return self._execute_with_recovery(
            self.config.validation_command,
            output_dir=output_dir,
            timeout_seconds=self.config.validation_timeout_seconds,
            values=values,
        )

    def _run_baseline(self, state: dict[str, Any]) -> None:
        if state.get("baseline"):
            return
        require_capacity(
            self.config,
            state,
            timeout_seconds=self.config.baseline_timeout_seconds + self.config.validation_timeout_seconds,
            consume_iteration=False,
        )
        baseline_dir = self.store.root / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        result_path = baseline_dir / "result.json"
        submission_path = baseline_dir / "submission.csv"
        values = self._render_values(
            result_path=str(result_path),
            submission_path=str(submission_path),
            experiment_id="official_baseline",
            strategy="official_baseline",
        )
        state["status"] = "baseline_running"
        self.store.save_state(state)
        self.store.append_event("baseline_started", {})
        attempts = self._execute_with_recovery(
            self.config.baseline_command,
            output_dir=baseline_dir / "execution",
            timeout_seconds=self.config.baseline_timeout_seconds,
            values=values,
        )
        if not result_path.is_file():
            raise ExecutionFailure("Official baseline did not create result JSON", failure_class="schema_alignment")
        result = read_json(result_path)
        metrics, score, resources = _score_from_result(self.config, result)
        for name, expected in self.config.baseline_expected_metrics.items():
            actual = score if name == "official_selection_score" else metrics.get(name)
            if actual is None or abs(float(actual) - expected) > self.config.baseline_tolerance:
                raise ContractError(
                    f"Official baseline mismatch for {name}: expected {expected}, got {actual} "
                    f"(tolerance {self.config.baseline_tolerance})"
                )
        validation_attempts = self._validate_submission(submission_path, baseline_dir / "validation")
        self._account_resources(
            state,
            resources=resources,
            attempts=attempts,
            validation_attempts=validation_attempts,
        )
        baseline_record = {
            "experiment_id": "official_baseline",
            "status": "reproduced",
            "metrics": metrics,
            "selection_score": score,
            "resources": resources,
            "submission_path": str(submission_path),
            "submission_sha256": sha256_file(submission_path),
            "attempts": attempts,
            "validation_attempts": validation_attempts,
            "completed_at": utc_timestamp(),
        }
        state["baseline"] = baseline_record
        state["incumbent"] = baseline_record.copy()
        state["best_selection_score"] = score
        state["status"] = "running"
        self.store.save_state(state)
        self.store.append_event("baseline_reproduced", {"metrics": metrics, "score": score})

    def _recover_interrupted_iteration(self, state: dict[str, Any]) -> None:
        active = state.get("active_experiment")
        if not active:
            return
        summary = {
            **active,
            "status": "failed",
            "failure_class": "interrupted",
            "completed_at": utc_timestamp(),
            "selection_score": None,
        }
        state.setdefault("experiments", []).append(summary)
        state["active_experiment"] = None
        self.store.save_state(state)
        self.store.append_event("interrupted_experiment_recovered", summary)

    def _run_experiment(self, state: dict[str, Any], decision: PlanDecision) -> None:
        experiment = decision.experiment
        require_capacity(
            self.config,
            state,
            timeout_seconds=experiment.timeout_seconds + self.config.validation_timeout_seconds,
            consume_iteration=True,
        )
        iteration = int(state["iterations_used"]) + 1
        experiment_dir = self.store.experiment_dir(iteration, experiment.experiment_id)
        result_path = experiment_dir / "result.json"
        submission_path = experiment_dir / "submission.csv"
        proposal = {
            "run_id": self.store.run_id,
            "benchmark": self.config.benchmark_name,
            "iteration": iteration,
            "experiment_id": experiment.experiment_id,
            "family": experiment.family,
            "hypothesis": experiment.hypothesis,
            "expected_effect": experiment.expected_effect,
            "knowledge_ids": list(experiment.knowledge_ids),
            "planner_reason": decision.reason,
            "planner_mode": decision.planner_mode,
            "evidence": list(decision.evidence),
            "success_rule": f"strictly improve {self.config.selection_metric} over incumbent",
            "falsification_rule": f"no improvement in {self.config.selection_metric}",
            "created_at": utc_timestamp(),
        }
        atomic_write_json(experiment_dir / "proposal.json", proposal)
        state["iterations_used"] = iteration
        state["active_experiment"] = {
            "iteration": iteration,
            "experiment_id": experiment.experiment_id,
            "family": experiment.family,
            "started_at": utc_timestamp(),
        }
        self.store.save_state(state)
        self.store.append_event("experiment_started", proposal)

        values = self._render_values(
            result_path=str(result_path),
            submission_path=str(submission_path),
            experiment_id=experiment.experiment_id,
            strategy=experiment.experiment_id,
        )
        attempts: list[dict[str, Any]] = []
        validation_attempts: list[dict[str, Any]] = []
        record: dict[str, Any]
        try:
            attempts = self._execute_with_recovery(
                experiment.command,
                output_dir=experiment_dir / "execution",
                timeout_seconds=experiment.timeout_seconds,
                values=values,
            )
            if not result_path.is_file():
                raise ExecutionFailure("Experiment did not create result JSON", failure_class="schema_alignment")
            metrics, score, resources = _score_from_result(self.config, read_json(result_path))
            validation_attempts = self._validate_submission(
                submission_path, experiment_dir / "validation"
            )
            self._account_resources(
                state,
                resources=resources,
                attempts=attempts,
                validation_attempts=validation_attempts,
            )
            incumbent_score = float(state["incumbent"]["selection_score"])
            promoted = _better(self.config, score, incumbent_score)
            status = "promoted" if promoted else "rejected"
            record = {
                **proposal,
                "status": status,
                "metrics": metrics,
                "selection_score": score,
                "resources": resources,
                "incumbent_score_before": incumbent_score,
                "submission_path": str(submission_path),
                "submission_sha256": sha256_file(submission_path),
                "attempts": attempts,
                "validation_attempts": validation_attempts,
                "completed_at": utc_timestamp(),
                "interpretation": (
                    "Hypothesis supported by the official validation selection rule"
                    if promoted
                    else "Hypothesis not supported by the official validation selection rule"
                ),
            }
            if promoted:
                state["incumbent"] = {
                    "experiment_id": experiment.experiment_id,
                    "status": "promoted",
                    "metrics": metrics,
                    "selection_score": score,
                    "submission_path": str(submission_path),
                    "submission_sha256": record["submission_sha256"],
                }
            update_convergence(self.config, state, score)
        except (ExecutionFailure, ContractError) as exc:
            failure_class = getattr(exc, "failure_class", "contract")
            attempts = getattr(exc, "attempts", attempts)
            record = {
                **proposal,
                "status": "timed_out" if failure_class == "timeout" else "failed",
                "failure_class": failure_class,
                "error": str(exc),
                "selection_score": None,
                "attempts": attempts,
                "validation_attempts": validation_attempts,
                "completed_at": utc_timestamp(),
                "interpretation": "Candidate execution was invalid; no promotion decision was made",
            }
            state["consecutive_small_improvements"] = int(
                state.get("consecutive_small_improvements", 0)
            ) + 1

        self.store.write_experiment_record(experiment_dir, record)
        state.setdefault("experiments", []).append(
            {
                "iteration": iteration,
                "experiment_id": experiment.experiment_id,
                "family": experiment.family,
                "status": record["status"],
                "selection_score": record.get("selection_score"),
                "record_path": str(experiment_dir / "record.json"),
                "failure_class": record.get("failure_class"),
            }
        )
        state["active_experiment"] = None
        self.store.save_state(state)
        self.store.append_event(
            "experiment_completed",
            {
                "iteration": iteration,
                "experiment_id": experiment.experiment_id,
                "status": record["status"],
                "selection_score": record.get("selection_score"),
            },
        )

    def _finalize(self, state: dict[str, Any], reason: str) -> None:
        incumbent = state.get("incumbent")
        if not incumbent:
            raise AgentError("Cannot finalize without a reproduced baseline or incumbent")
        source = Path(incumbent["submission_path"])
        if not source.is_file():
            raise AgentError(f"Incumbent submission is missing: {source}")
        final_submission = self.store.root / "final_submission.csv"
        shutil.copy2(source, final_submission)
        validation_attempts = self._validate_submission(
            final_submission, self.store.root / "final_validation"
        )
        final = {
            "designated_at": utc_timestamp(),
            "stop_reason": reason,
            "experiment_id": incumbent["experiment_id"],
            "metrics": incumbent["metrics"],
            "selection_score": incumbent["selection_score"],
            "submission_path": str(final_submission),
            "submission_sha256": sha256_file(final_submission),
            "validation_attempts": validation_attempts,
            "hidden_test_evaluations": 0,
            "iterations_used": state["iterations_used"],
            "manual_interventions": len(state.get("manual_interventions", [])),
            "resource_usage": state.get("resources", {}),
            "elapsed_seconds": snapshot(self.config, state).elapsed_seconds,
        }
        atomic_write_json(self.store.root / "final_manifest.json", final)
        state["final"] = final
        state["status"] = "completed"
        state["stop_reason"] = reason
        self.store.save_state(state)
        self.store.append_event("run_completed", final)

    def execute(self, *, resume: bool = False) -> dict[str, Any]:
        try:
            self.contract_report = validate_contract(self.config)
        except ContractError as exc:
            if resume and self.store.state_path.exists():
                state = self.store.load_state()
            elif not self.store.root.exists():
                state = self.store.create()
            else:
                state = self.store.load_state()
            state["status"] = "blocked_contract"
            state["stop_reason"] = str(exc)
            self.store.save_state(state)
            self.store.append_event("contract_blocked", {"error": str(exc)})
            raise

        if resume:
            state = self.store.load_state()
            if state.get("status") == "completed":
                return state
            self._recover_interrupted_iteration(state)
            self.store.append_event("run_resumed", {})
        else:
            state = self.store.create()
        atomic_write_json(self.store.root / "contract_report.json", self.contract_report)

        try:
            self._run_baseline(state)
            while True:
                current = snapshot(self.config, state)
                if converged(self.config, state):
                    reason = "converged"
                    break
                if current.iterations_remaining <= 0:
                    reason = "iteration_cap"
                    break
                if current.remaining_seconds <= self.config.validation_timeout_seconds:
                    reason = "wall_clock_budget"
                    break
                decision = self.planner.choose(self.config, state)
                if decision is None:
                    reason = "experiment_catalog_exhausted"
                    break
                state["resources"]["command_seconds"] = float(
                    state["resources"].get("command_seconds", 0.0)
                ) + decision.planner_command_seconds
                state["resources"]["llm_tokens"] = int(
                    state["resources"].get("llm_tokens", 0)
                ) + decision.planner_llm_tokens
                self.store.save_state(state)
                try:
                    self._run_experiment(state, decision)
                except BudgetError:
                    reason = "insufficient_time_for_next_experiment"
                    break
            self._finalize(state, reason)
            return state
        except (AgentError, OSError, ValueError) as exc:
            state["status"] = "failed"
            state["stop_reason"] = str(exc)
            self.store.save_state(state)
            self.store.append_event("run_failed", {"error": str(exc), "type": type(exc).__name__})
            raise
