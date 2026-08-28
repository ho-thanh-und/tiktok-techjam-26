from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .contracts import validate_contract
from .errors import AgentError
from .orchestrator import AutonomousRun
from .dashboard import serve
from .env_file import load_env_file
from .reporting import write_report
from .storage import RunStore


def _new_run_id(benchmark: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in benchmark).strip("-").lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe}-{stamp}"


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automl-agent")
    parser.add_argument("--config", required=True, help="Path to an agent JSON configuration")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Validate the benchmark contract and assets")
    run = sub.add_parser("run", help="Start a new autonomous benchmark run")
    run.add_argument("--run-id", help="Stable run identifier")
    resume = sub.add_parser("resume", help="Resume an interrupted run")
    resume.add_argument("run_id")
    status = sub.add_parser("status", help="Show persistent run state")
    status.add_argument("run_id")
    report = sub.add_parser("report", help="Generate a Markdown report for a run")
    report.add_argument("run_id")
    report.add_argument("--output")
    dashboard = sub.add_parser("serve", help="Serve the read-only local run dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command in {"preflight", "run", "resume"} and config.planner_env_file is not None:
            load_env_file(config.planner_env_file)
        if args.command == "preflight":
            _print_json(validate_contract(config))
            return 0
        if args.command == "status":
            _print_json(RunStore(config, args.run_id).load_state())
            return 0
        if args.command == "report":
            output = Path(args.output).resolve() if args.output else None
            print(write_report(config.run_root, args.run_id, output))
            return 0
        if args.command == "serve":
            serve(config.run_root, args.host, args.port)
            return 0
        if args.command == "run":
            run_id = args.run_id or _new_run_id(config.benchmark_name)
            state = AutonomousRun(config, RunStore(config, run_id)).execute()
            _print_json(state)
            return 0
        if args.command == "resume":
            state = AutonomousRun(config, RunStore(config, args.run_id)).execute(resume=True)
            _print_json(state)
            return 0
        raise AssertionError("unreachable")
    except (AgentError, OSError, ValueError) as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2
