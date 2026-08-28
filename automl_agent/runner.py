from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ExecutionFailure
from .io_utils import atomic_write_json, utc_timestamp


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path


def render_command(command: tuple[str, ...], values: dict[str, str]) -> tuple[str, ...]:
    rendered: list[str] = []
    for argument in command:
        try:
            rendered.append(argument.format_map(values))
        except KeyError as exc:
            raise ExecutionFailure(f"Unknown command placeholder {exc} in {argument!r}") from exc
    return tuple(rendered)


def classify_failure(exit_code: int, stderr: str) -> str:
    lowered = stderr.lower()
    if "memoryerror" in lowered or "out of memory" in lowered or "cuda oom" in lowered:
        return "out_of_memory"
    if "no such file" in lowered or "cannot find" in lowered or "filenotfound" in lowered:
        return "missing_input"
    if "nan" in lowered or "diverg" in lowered:
        return "numerical"
    if exit_code in {75, 111} or "temporar" in lowered or "connection reset" in lowered:
        return "transient"
    if "schema" in lowered or "alignment" in lowered:
        return "schema_alignment"
    return "command_failed"


def run_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    output_dir: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> CommandResult:
    if not argv:
        raise ExecutionFailure("Cannot execute an empty command")
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    heartbeat_path = output_dir / "heartbeat.json"
    started = time.monotonic()

    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            shell=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            text=True,
        )
        last_heartbeat = 0.0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed - last_heartbeat >= max(1.0, poll_seconds):
                atomic_write_json(
                    heartbeat_path,
                    {"at": utc_timestamp(), "pid": process.pid, "elapsed_seconds": elapsed},
                )
                last_heartbeat = elapsed
            if elapsed >= timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise ExecutionFailure(
                    f"Command timed out after {elapsed:.1f}s: {argv!r}",
                    failure_class="timeout",
                )
            time.sleep(min(max(poll_seconds, 0.05), 1.0))
        exit_code = int(process.returncode)

    elapsed = time.monotonic() - started
    if exit_code != 0:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise ExecutionFailure(
            f"Command exited with {exit_code}: {argv!r}",
            failure_class=classify_failure(exit_code, stderr_tail),
        )
    return CommandResult(argv, exit_code, elapsed, stdout_path, stderr_path)

