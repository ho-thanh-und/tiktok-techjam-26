from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .errors import ContractError


ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _parse_value(raw: str, *, path: Path, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid double-quoted value in {path} line {line_number}") from exc
        if not isinstance(parsed, str):
            raise ContractError(f"Invalid value in {path} line {line_number}")
        return parsed
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ContractError(f"Invalid single-quoted value in {path} line {line_number}")
        return value[1:-1]
    return value


def load_env_file(path: Path, *, override: bool = False) -> tuple[str, ...]:
    """Load a small dotenv subset without ever returning or logging secret values."""
    if not path.exists():
        return ()
    if not path.is_file():
        raise ContractError(f"Environment file is not a regular file: {path}")
    loaded: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ContractError(f"Invalid environment assignment in {path} line {line_number}")
            name, raw_value = line.split("=", 1)
            name = name.strip()
            if not ENV_NAME.fullmatch(name):
                raise ContractError(f"Invalid environment name in {path} line {line_number}")
            value = _parse_value(raw_value, path=path, line_number=line_number)
            if override or name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return tuple(loaded)
