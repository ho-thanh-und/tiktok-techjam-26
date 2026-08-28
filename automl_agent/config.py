from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError
from .io_utils import read_json


@dataclass(frozen=True)
class Asset:
    role: str
    path: Path
    kind: str = "file"
    sha256: str | None = None


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    family: str
    priority: int
    hypothesis: str
    knowledge_ids: tuple[str, ...]
    command: tuple[str, ...]
    timeout_seconds: int
    expected_effect: str


@dataclass(frozen=True)
class AgentConfig:
    config_path: Path
    workspace: Path
    run_root: Path
    benchmark_name: str
    profile: str
    positive_label: str
    metric_names: tuple[str, ...]
    selection_metric: str
    maximize: bool
    iteration_cap: int
    wall_clock_seconds: int
    epsilon: float
    convergence_patience: int
    command_poll_seconds: float
    assets: tuple[Asset, ...]
    forbidden_paths: tuple[Path, ...]
    baseline_command: tuple[str, ...]
    baseline_timeout_seconds: int
    baseline_expected_metrics: dict[str, float]
    baseline_tolerance: float
    validation_command: tuple[str, ...]
    validation_timeout_seconds: int
    experiments: tuple[Experiment, ...]
    retry_transient_once: bool
    planner_mode: str
    planner_command: tuple[str, ...] | None
    planner_timeout_seconds: int
    planner_fallback_to_catalog: bool
    planner_provider: str | None
    planner_model: str | None
    planner_api_key_env: str | None
    planner_env_file: Path | None


def _require(mapping: dict[str, Any], key: str, expected_type: type | tuple[type, ...]) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_name = " or ".join(item.__name__ for item in expected_type)
        else:
            expected_name = expected_type.__name__
        raise ContractError(f"Config key {key!r} must be {expected_name}")
    return value


def _command(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ContractError(f"{name} must be a non-empty JSON array of strings")
    return tuple(value)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> AgentConfig:
    config_path = Path(path).resolve()
    raw = read_json(config_path)
    if raw.get("schema_version") != 1:
        raise ContractError("Only config schema_version 1 is supported")

    workspace_value = raw.get("workspace", ".")
    if not isinstance(workspace_value, str):
        raise ContractError("workspace must be a string path")
    workspace = _resolve(config_path.parent, workspace_value)
    run_root_value = raw.get("run_root", "artifacts/agent_runs")
    if not isinstance(run_root_value, str):
        raise ContractError("run_root must be a string path")
    run_root = _resolve(workspace, run_root_value)

    benchmark = _require(raw, "benchmark", dict)
    metrics = _require(raw, "metrics", dict)
    budget = _require(raw, "budget", dict)
    convergence = _require(raw, "convergence", dict)
    baseline = _require(raw, "official_baseline", dict)
    submission = _require(raw, "submission", dict)
    planner = raw.get("planner", {"mode": "catalog"})
    if not isinstance(planner, dict):
        raise ContractError("planner must be an object")
    planner_mode = str(planner.get("mode", "catalog"))
    if planner_mode not in {"catalog", "command", "llm"}:
        raise ContractError("planner.mode must be catalog, command, or llm")
    planner_command = None
    planner_provider = None
    planner_model = None
    planner_api_key_env = None
    planner_env_file = None
    if planner_mode == "command":
        planner_command = _command(planner.get("command"), "planner.command")
    elif planner_mode == "llm":
        provider = str(planner.get("provider", "openai"))
        if provider not in {"openai", "gemini"}:
            raise ContractError("planner.provider must be openai or gemini")
        default_model = "gemini-3.7-flash" if provider == "gemini" else "gpt-5-mini"
        model = str(planner.get("model", default_model))
        if not model.strip():
            raise ContractError("planner.model must be a non-empty string")
        default_base_url = (
            "https://generativelanguage.googleapis.com/v1beta"
            if provider == "gemini"
            else "https://api.openai.com/v1"
        )
        base_url = str(planner.get("base_url", default_base_url))
        if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ContractError("planner.base_url must use HTTPS or a loopback HTTP address")
        default_key_env = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        api_key_env = str(planner.get("api_key_env", default_key_env))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ContractError("planner.api_key_env must be a valid environment-variable name")
        env_file_value = planner.get("env_file", ".env")
        if not isinstance(env_file_value, str) or not env_file_value:
            raise ContractError("planner.env_file must be a non-empty path string")
        planner_env_file = _resolve(workspace, env_file_value)
        api_timeout_seconds = float(planner.get("api_timeout_seconds", 60))
        max_output_tokens = int(planner.get("max_output_tokens", 1200))
        if api_timeout_seconds <= 0:
            raise ContractError("planner.api_timeout_seconds must be positive")
        if max_output_tokens <= 0:
            raise ContractError("planner.max_output_tokens must be positive")
        planner_provider = provider
        planner_model = model
        planner_api_key_env = api_key_env
        planner_command = (
            "{python}",
            "-m",
            "automl_agent.llm_planner_cli",
            "--provider",
            provider,
            "--evidence",
            "{evidence_path}",
            "--decision",
            "{decision_path}",
            "--model",
            model,
            "--base-url",
            base_url,
            "--api-key-env",
            api_key_env,
            "--env-file",
            str(planner_env_file),
            "--api-timeout-seconds",
            str(api_timeout_seconds),
            "--max-output-tokens",
            str(max_output_tokens),
        )

    names = metrics.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ContractError("metrics.names must be a non-empty string array")

    assets_raw = raw.get("assets", [])
    if not isinstance(assets_raw, list):
        raise ContractError("assets must be an array")
    assets: list[Asset] = []
    for item in assets_raw:
        if not isinstance(item, dict):
            raise ContractError("Each asset must be an object")
        role = _require(item, "role", str)
        asset_path = _resolve(workspace, _require(item, "path", str))
        digest = item.get("sha256")
        if digest is not None and not isinstance(digest, str):
            raise ContractError(f"Asset sha256 for {role} must be a string")
        kind = str(item.get("kind", "file"))
        if kind not in {"file", "directory"}:
            raise ContractError(f"Asset kind for {role} must be file or directory")
        assets.append(Asset(role=role, path=asset_path, kind=kind, sha256=digest))

    forbidden_raw = raw.get("forbidden_paths", [])
    if not isinstance(forbidden_raw, list) or not all(isinstance(x, str) for x in forbidden_raw):
        raise ContractError("forbidden_paths must be an array of paths")

    experiments_raw = raw.get("experiments", [])
    if not isinstance(experiments_raw, list):
        raise ContractError("experiments must be an array")
    experiments: list[Experiment] = []
    for item in experiments_raw:
        if not isinstance(item, dict):
            raise ContractError("Each experiment must be an object")
        knowledge_ids = item.get("knowledge_ids", [])
        if not isinstance(knowledge_ids, list) or not all(isinstance(x, str) for x in knowledge_ids):
            raise ContractError("experiment knowledge_ids must be a string array")
        experiments.append(
            Experiment(
                experiment_id=_require(item, "id", str),
                family=_require(item, "family", str),
                priority=int(item.get("priority", 0)),
                hypothesis=_require(item, "hypothesis", str),
                knowledge_ids=tuple(knowledge_ids),
                command=_command(item.get("command"), f"experiment {item.get('id')} command"),
                timeout_seconds=int(item.get("timeout_seconds", 600)),
                expected_effect=str(item.get("expected_effect", "unspecified")),
            )
        )

    expected = baseline.get("expected_metrics", {})
    if not isinstance(expected, dict) or not all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in expected.items()
    ):
        raise ContractError("official_baseline.expected_metrics must be numeric")

    return AgentConfig(
        config_path=config_path,
        workspace=workspace,
        run_root=run_root,
        benchmark_name=_require(benchmark, "name", str),
        profile=str(benchmark.get("profile", "competition")),
        positive_label=_require(benchmark, "positive_label", str),
        metric_names=tuple(names),
        selection_metric=_require(metrics, "selection", str),
        maximize=bool(metrics.get("maximize", True)),
        iteration_cap=int(_require(budget, "iterations", int)),
        wall_clock_seconds=int(_require(budget, "wall_clock_seconds", int)),
        epsilon=float(_require(convergence, "epsilon", (int, float))),
        convergence_patience=int(_require(convergence, "patience", int)),
        command_poll_seconds=float(raw.get("command_poll_seconds", 0.25)),
        assets=tuple(assets),
        forbidden_paths=tuple(_resolve(workspace, x) for x in forbidden_raw),
        baseline_command=_command(baseline.get("command"), "official_baseline.command"),
        baseline_timeout_seconds=int(baseline.get("timeout_seconds", 1800)),
        baseline_expected_metrics={k: float(v) for k, v in expected.items()},
        baseline_tolerance=float(baseline.get("tolerance", 1e-6)),
        validation_command=_command(submission.get("validation_command"), "submission.validation_command"),
        validation_timeout_seconds=int(submission.get("timeout_seconds", 300)),
        experiments=tuple(experiments),
        retry_transient_once=bool(raw.get("recovery", {}).get("retry_transient_once", True)),
        planner_mode=planner_mode,
        planner_command=planner_command,
        planner_timeout_seconds=int(planner.get("timeout_seconds", 120)),
        planner_fallback_to_catalog=bool(planner.get("fallback_to_catalog", True)),
        planner_provider=planner_provider,
        planner_model=planner_model,
        planner_api_key_env=planner_api_key_env,
        planner_env_file=planner_env_file,
    )


def placeholders(config: AgentConfig) -> dict[str, str]:
    return {
        "python": sys.executable,
        "workspace": str(config.workspace),
        "config_dir": str(config.config_path.parent),
    }
