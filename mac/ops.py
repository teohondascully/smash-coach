"""Mac-side ops module: pod lifecycle, metrics polling, and cost tracking.

Three classes:
  - PodLifecycle:  subprocess wrapper around the `prime` CLI.
  - PodMetrics:    async httpx client for the pod-side ops agent.
  - CostTracker:   persists started_at to disk and computes spend/burn.

Lazy imports keep this file importable even where heavy deps are absent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from typing import Any

CLI_UNAVAILABLE = "prime CLI not found"


# ---------------------------------------------------------------------------
# PodLifecycle
# ---------------------------------------------------------------------------

# Default templates. Edit here as the `prime` CLI surface changes.
START_CMD = ["prime", "pods", "create", "--gpu-type", "H100", "--count", "8"]
STOP_CMD = ["prime", "pods", "stop"]  # appends pod_id
LIST_CMD = ["prime", "pods", "list"]
SSH_CMD = ["prime", "pods", "ssh"]  # appends pod_id


class PodLifecycle:
    """Subprocess wrapper around the `prime` CLI.

    All subprocess calls return either parsed output, a sentinel string when
    the CLI is missing, or None/False on failure.
    """

    def __init__(
        self,
        start_cmd: list[str] | None = None,
        stop_cmd: list[str] | None = None,
        list_cmd: list[str] | None = None,
        ssh_cmd: list[str] | None = None,
    ) -> None:
        self.start_cmd = list(start_cmd or START_CMD)
        self.stop_cmd = list(stop_cmd or STOP_CMD)
        self.list_cmd = list(list_cmd or LIST_CMD)
        self.ssh_cmd = list(ssh_cmd or SSH_CMD)

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str] | str:
        try:
            return subprocess.run(
                argv, capture_output=True, timeout=30, text=True, check=False
            )
        except FileNotFoundError:
            return CLI_UNAVAILABLE
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(argv, returncode=124, stdout="", stderr="timeout")

    @staticmethod
    def _parse_pod_id(text: str) -> str | None:
        """Best-effort extraction of a pod id from `prime pods create` stdout.

        Heuristics in order:
          1. JSON containing an "id" field.
          2. Lines like "Pod id: abc123" or "pod_id=abc123".
          3. A hex-ish 8+ char token.
        """
        text = text.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ("id", "pod_id", "podId"):
                    if key in obj and isinstance(obj[key], str):
                        return obj[key]
        except json.JSONDecodeError:
            pass

        m = re.search(r"(?:pod[_\s-]?id|id)\s*[:=]\s*([A-Za-z0-9_\-]+)", text, re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.search(r"\b([a-f0-9]{8,})\b", text)
        if m:
            return m.group(1)
        return None

    def start(self) -> str | None:
        res = self._run(self.start_cmd)
        if isinstance(res, str):
            return None
        if res.returncode != 0:
            return None
        return self._parse_pod_id(res.stdout)

    def stop(self, pod_id: str) -> bool:
        res = self._run([*self.stop_cmd, pod_id])
        if isinstance(res, str):
            return False
        return res.returncode == 0

    def list(self) -> list[dict[str, Any]]:
        # Try JSON output first.
        for extra in (["--json"], ["--format=json"], []):
            res = self._run([*self.list_cmd, *extra])
            if isinstance(res, str):
                return []
            if res.returncode != 0:
                continue
            stdout = res.stdout.strip()
            if not stdout:
                continue
            try:
                obj = json.loads(stdout)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, dict)]
                if isinstance(obj, dict):
                    for key in ("pods", "data", "items"):
                        if key in obj and isinstance(obj[key], list):
                            return [x for x in obj[key] if isinstance(x, dict)]
            except json.JSONDecodeError:
                # Fall through to text parsing.
                pass

            # Plain text parser: assume first line is a header, columns are
            # whitespace-separated. Look for an "ID" column.
            lines = [line for line in stdout.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            header = re.split(r"\s{2,}|\t", lines[0].strip())
            pods: list[dict[str, Any]] = []
            for line in lines[1:]:
                fields = re.split(r"\s{2,}|\t", line.strip())
                row: dict[str, Any] = {}
                for i, name in enumerate(header):
                    row[name.lower()] = fields[i] if i < len(fields) else ""
                if not row.get("id"):
                    # Fallback: first token as id.
                    tokens = line.split()
                    if tokens:
                        row["id"] = tokens[0]
                pods.append(row)
            return pods
        return []

    def restart_service(self, pod_id: str, service: str) -> bool:
        """Attempt to restart a systemd service on the pod via `prime pods ssh`.

        Many `prime` CLIs do not support an inline-command form. If the call
        fails for any reason we return False and let the caller surface a
        "manual restart required" message.
        """
        cmd = [*self.ssh_cmd, pod_id, f"sudo systemctl restart {service}"]
        res = self._run(cmd)
        if isinstance(res, str):
            return False
        return res.returncode == 0


# ---------------------------------------------------------------------------
# PodMetrics
# ---------------------------------------------------------------------------


class PodMetrics:
    """Async client polling the pod-side ops agent."""

    def __init__(self, agent_url: str, timeout_s: float = 2.0) -> None:
        self.agent_url = agent_url.rstrip("/")
        self.timeout_s = timeout_s

    async def _get(self, path: str) -> Any:
        import httpx  # lazy

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.agent_url}{path}")
            r.raise_for_status()
            return r.json()

    async def gpu(self) -> list[dict] | None:
        try:
            data = await self._get("/gpu")
            return data if isinstance(data, list) else None
        except Exception:
            return None

    async def health_s1(self) -> dict | None:
        try:
            data = await self._get("/health/s1")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def health_s2(self) -> dict | None:
        try:
            data = await self._get("/health/s2")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def disk(self) -> dict | None:
        try:
            data = await self._get("/disk")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def snapshot(self) -> dict:
        """Fetch all four endpoints concurrently.

        Each field is None on failure. `fetched_at` is epoch seconds.
        """
        results = await asyncio.gather(
            self.gpu(),
            self.health_s1(),
            self.health_s2(),
            self.disk(),
            return_exceptions=True,
        )

        def _ok(x: Any) -> Any:
            return None if isinstance(x, BaseException) else x

        return {
            "gpu": _ok(results[0]),
            "s1": _ok(results[1]),
            "s2": _ok(results[2]),
            "disk": _ok(results[3]),
            "fetched_at": time.time(),
        }


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Persists pod start time and computes runtime / spend / burn.

    State file format:
        {"started_at": <epoch_seconds or null>}
    """

    def __init__(self, state_path: str, rate_per_hour: float, budget_usd: float) -> None:
        self.state_path = state_path
        self.rate_per_hour = float(rate_per_hour)
        self.budget_usd = float(budget_usd)

    def _read(self) -> dict[str, Any]:
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, obj: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, self.state_path)

    def mark_started(self, when: float | None = None) -> None:
        self._write({"started_at": when if when is not None else time.time()})

    def mark_stopped(self) -> None:
        self._write({"started_at": None})

    def _started_at(self) -> float | None:
        v = self._read().get("started_at")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def runtime_seconds(self, now: float | None = None) -> float:
        started = self._started_at()
        if started is None:
            return 0.0
        now = now if now is not None else time.time()
        return max(0.0, now - started)

    def spent_usd(self, now: float | None = None) -> float:
        hours = self.runtime_seconds(now) / 3600.0
        return round(hours * self.rate_per_hour, 4)

    def remaining_usd(self, now: float | None = None) -> float:
        return round(self.budget_usd - self.spent_usd(now), 4)

    def burn_rate_per_hour(self) -> float:
        return self.rate_per_hour if self._started_at() is not None else 0.0
