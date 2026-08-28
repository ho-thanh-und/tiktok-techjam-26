from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .errors import BudgetError


@dataclass(frozen=True)
class BudgetSnapshot:
    iterations_used: int
    iterations_remaining: int
    elapsed_seconds: float
    remaining_seconds: float


def snapshot(config: AgentConfig, state: dict[str, Any], now: float | None = None) -> BudgetSnapshot:
    current = time.time() if now is None else now
    started = float(state["started_at_epoch"])
    elapsed = max(0.0, current - started)
    used = int(state.get("iterations_used", 0))
    return BudgetSnapshot(
        iterations_used=used,
        iterations_remaining=max(0, config.iteration_cap - used),
        elapsed_seconds=elapsed,
        remaining_seconds=max(0.0, config.wall_clock_seconds - elapsed),
    )


def require_capacity(
    config: AgentConfig,
    state: dict[str, Any],
    *,
    timeout_seconds: int,
    consume_iteration: bool,
) -> BudgetSnapshot:
    current = snapshot(config, state)
    if consume_iteration and current.iterations_remaining <= 0:
        raise BudgetError("Iteration cap reached")
    if current.remaining_seconds <= 0:
        raise BudgetError("Wall-clock budget reached")
    if timeout_seconds > current.remaining_seconds:
        raise BudgetError(
            f"Action timeout {timeout_seconds}s exceeds remaining wall time "
            f"{current.remaining_seconds:.1f}s"
        )
    return current


def update_convergence(config: AgentConfig, state: dict[str, Any], score: float) -> None:
    best_before = state.get("best_selection_score")
    materially_improved = best_before is None or (
        score > float(best_before) + config.epsilon
        if config.maximize
        else score < float(best_before) - config.epsilon
    )
    is_new_best = best_before is None or (
        score > float(best_before) if config.maximize else score < float(best_before)
    )
    if is_new_best:
        state["best_selection_score"] = score
    if materially_improved:
        state["consecutive_small_improvements"] = 0
    else:
        state["consecutive_small_improvements"] = int(
            state.get("consecutive_small_improvements", 0)
        ) + 1


def converged(config: AgentConfig, state: dict[str, Any]) -> bool:
    return int(state.get("consecutive_small_improvements", 0)) >= config.convergence_patience
