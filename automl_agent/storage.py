from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .errors import AgentError
from .io_utils import atomic_write_json, read_json, utc_timestamp


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RunStore:
    def __init__(self, config: AgentConfig, run_id: str) -> None:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise AgentError(f"Unsafe run ID: {run_id!r}")
        self.config = config
        self.run_id = run_id
        self.root = (config.run_root / run_id).resolve()
        try:
            self.root.relative_to(config.run_root.resolve())
        except ValueError as exc:
            raise AgentError("Run path escapes configured run root") from exc
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.experiments_dir = self.root / "experiments"

    def create(self) -> dict[str, Any]:
        if self.root.exists():
            raise AgentError(f"Run already exists: {self.run_id}")
        self.experiments_dir.mkdir(parents=True)
        now = time.time()
        state: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "benchmark": self.config.benchmark_name,
            "profile": self.config.profile,
            "status": "created",
            "started_at": utc_timestamp(),
            "started_at_epoch": now,
            "updated_at": utc_timestamp(),
            "iterations_used": 0,
            "consecutive_small_improvements": 0,
            "best_selection_score": None,
            "baseline": None,
            "incumbent": None,
            "experiments": [],
            "manual_interventions": [],
            "resources": {
                "command_seconds": 0.0,
                "gpu_hours": 0.0,
                "llm_tokens": 0,
            },
            "stop_reason": None,
            "final": None,
        }
        self.save_state(state)
        self.append_event("run_created", {"config": str(self.config.config_path)})
        return state

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise AgentError(f"Run state does not exist: {self.state_path}")
        return read_json(self.state_path)

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_timestamp()
        atomic_write_json(self.state_path, state)

    def append_event(self, event: str, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"at": utc_timestamp(), "event": event, "payload": payload}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def experiment_dir(self, iteration: int, experiment_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_id)
        path = self.experiments_dir / f"{iteration:02d}-{safe}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_experiment_record(self, path: Path, record: dict[str, Any]) -> None:
        record_path = path / "record.json"
        if record_path.exists() and read_json(record_path).get("status") in {
            "succeeded",
            "failed",
            "timed_out",
            "rejected",
            "promoted",
        }:
            raise AgentError(f"Refusing to overwrite terminal experiment record: {record_path}")
        atomic_write_json(record_path, record)
