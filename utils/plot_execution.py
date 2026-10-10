# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Trusted Python plot execution, with a bounded, terminable process helper.

These helpers execute arbitrary Python. Neither a process nor a timeout is a
security sandbox. Only run code from a trusted source in a trusted single-user
installation. Use the scientific-data plot backend for untrusted inputs.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import weakref
from typing import Optional


class PlotExecutionTimeout(TimeoutError):
    """The trusted plot worker exceeded its execution deadline and was killed."""


class LegacyExecutorHandle:
    """Compatibility lifecycle handle; allocates no pool and executes no work."""
    def shutdown(self, wait=True, **kwargs):
        pass


_LIMITERS = weakref.WeakKeyDictionary()


def extract_plot_code(code_text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", code_text, re.DOTALL)
    return match.group(1).strip() if match else code_text.strip()


def execute_plot_code_worker(code_text: str, dpi: int = 300) -> Optional[str]:
    """Low-level legacy entry point: execute trusted code in this process.

    Prefer execute_plot_code_async for generated code: this direct entry point
    cannot terminate a hung caller. It remains public for legacy integrations.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.rcdefaults()
    try:
        exec(extract_plot_code(code_text), {})
        if not plt.get_fignums():
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg", bbox_inches="tight", dpi=dpi)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Do not print arbitrary exception messages containing file/secret data.
        return None
    finally:
        plt.close("all")


async def _kill_worker(process):
    if process.returncode is None or os.name == "posix":
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


async def execute_plot_code_async(code_text: str, dpi: int = 300, *,
                                  timeout: float = 30.0, allow_execution: bool = False) -> Optional[str]:
    """Execute explicitly trusted code in a bounded child process.

    There are at most four children per event loop. Timeout/cancellation kills
    the worker process group on POSIX and waits for the worker to exit. This is
    resource/lifecycle control, not isolation from files, networks, or secrets.
    """
    if not allow_execution:
        raise PermissionError("Generated Python plot execution requires explicit trusted-code permission; this is not a sandbox")
    if not isinstance(code_text, str) or not code_text.strip():
        raise ValueError("Plot code must be nonempty text")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 1 <= dpi <= 1200:
        raise ValueError("Plot dpi must be an integer between 1 and 1200")
    import math
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Plot timeout must be positive and finite")
    loop = asyncio.get_running_loop()
    limiter = _LIMITERS.setdefault(loop, asyncio.Semaphore(4))
    async with limiter:
        with tempfile.TemporaryDirectory(prefix="paperbanana-trusted-plot-") as directory:
            output = Path(directory) / "result.json"
            environment = {key: value for key, value in os.environ.items()
                           if key in ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL")}
            environment["MPLCONFIGDIR"] = directory
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__).resolve()), "--trusted-worker", str(output),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL, cwd=directory, env=environment,
                start_new_session=os.name == "posix")
            try:
                payload = json.dumps({"code": code_text, "dpi": dpi}).encode("utf-8")
                await asyncio.wait_for(process.communicate(payload), timeout=timeout)
                if process.returncode != 0 or not output.is_file() or output.stat().st_size > 64 * 1024 * 1024:
                    return None
                value = json.loads(output.read_text(encoding="utf-8"))
                return value if isinstance(value, str) else None
            except asyncio.TimeoutError as exc:
                await _kill_worker(process)
                raise PlotExecutionTimeout("Trusted plot worker timed out and was terminated") from exc
            except asyncio.CancelledError:
                await _kill_worker(process)
                raise
            finally:
                await _kill_worker(process)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--trusted-worker":
        raise SystemExit(2)
    request = json.loads(sys.stdin.buffer.read())
    Path(sys.argv[2]).write_text(json.dumps(execute_plot_code_worker(request["code"], request["dpi"])), encoding="utf-8")
