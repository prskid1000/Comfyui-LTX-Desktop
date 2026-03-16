"""Manages the ComfyUI subprocess lifecycle — auto-start, health check, stop.

Startup runs in a background thread so it never blocks the API server.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Expected layout: LTX-Desktop and ComfyUI are siblings under the same root.
#   <root>/ComfyUI/       — ComfyUI installation
#   <root>/.venv/         — Python venv with ComfyUI dependencies
#   <root>/LTX-Desktop/   — This repo
# Override with env vars: COMFYUI_DIR, COMFYUI_VENV_PYTHON
_COMFYUI_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", str(_COMFYUI_ROOT / "ComfyUI")))
_COMFYUI_MAIN = _COMFYUI_DIR / "main.py"

# Discover the venv Python that has all ComfyUI dependencies.
_env_python = os.environ.get("COMFYUI_VENV_PYTHON")
if _env_python:
    _VENV_PYTHON = Path(_env_python)
else:
    _VENV_PYTHON = _COMFYUI_ROOT / ".venv" / "Scripts" / "python.exe"
    if not _VENV_PYTHON.exists():
        _VENV_PYTHON = _COMFYUI_ROOT / ".venv" / "bin" / "python"

_DEFAULT_ARGS = [
    "--listen",
    "--async-offload", "16",
    "--cache-none",
    "--disable-smart-memory",
    "--reserve-vram", "0.3",
]

# How long to wait for ComfyUI to become ready (seconds).
_STARTUP_TIMEOUT = 120
_HEALTH_POLL_INTERVAL = 3


class ComfyUIProcess:
    """Singleton-style manager for the ComfyUI subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._url: str = "http://127.0.0.1:8188"
        self._starting: bool = False
        self._start_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._start_failed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_running(self, url: str = "http://127.0.0.1:8188") -> bool:
        """Make sure ComfyUI is reachable. Start it in a background thread if not.

        Returns True if ComfyUI is already available.
        Returns False if startup was kicked off (call wait_until_ready to block).
        """
        self._url = url

        if self._is_healthy():
            self._ready_event.set()
            return True

        with self._start_lock:
            # Double-check after acquiring lock
            if self._is_healthy():
                self._ready_event.set()
                return True

            if self._starting:
                # Already being started by another thread
                return False

            self._starting = True
            self._start_failed = False
            self._ready_event.clear()

        # Launch startup in background thread
        t = threading.Thread(target=self._background_start, daemon=True, name="comfyui-starter")
        t.start()
        return False

    def ensure_running_blocking(self, url: str = "http://127.0.0.1:8188", timeout: float = 150) -> bool:
        """Start ComfyUI if needed and block until it's ready.

        Returns True if ComfyUI is available, False on failure/timeout.
        """
        if self.ensure_running(url):
            return True
        return self.wait_until_ready(timeout=timeout)

    def wait_until_ready(self, timeout: float = 150) -> bool:
        """Block until ComfyUI is ready (or startup fails/times out)."""
        if self._ready_event.wait(timeout=timeout):
            return not self._start_failed
        return False

    def stop(self) -> None:
        """Terminate the ComfyUI subprocess if we own it."""
        if self._process is None:
            return
        logger.info("Stopping ComfyUI process (pid %s)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def is_running(self) -> bool:
        """True if ComfyUI is responding to HTTP requests."""
        return self._is_healthy()

    @property
    def pid(self) -> int | None:
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _background_start(self) -> None:
        """Runs in a background thread: start process and wait for health."""
        try:
            # Clean up dead process
            if self._process is not None and self._process.poll() is not None:
                logger.warning("ComfyUI process exited (code %s) — restarting", self._process.returncode)
                self._process = None

            if self._process is None:
                if not self._start():
                    self._start_failed = True
                    return

            if self._wait_until_healthy():
                self._start_failed = False
            else:
                self._start_failed = True
        except Exception:
            logger.exception("ComfyUI background start failed")
            self._start_failed = True
        finally:
            self._starting = False
            self._ready_event.set()  # Unblock waiters regardless of outcome

    def _is_healthy(self) -> bool:
        try:
            r = requests.get(self._url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _start(self) -> bool:
        python = str(_VENV_PYTHON)
        main_py = str(_COMFYUI_MAIN)

        if not Path(python).exists():
            logger.error("ComfyUI venv Python not found at %s", python)
            return False
        if not Path(main_py).exists():
            logger.error("ComfyUI main.py not found at %s", main_py)
            return False

        cmd = [python, main_py, *_DEFAULT_ARGS]
        logger.info("Starting ComfyUI: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(_COMFYUI_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # On Windows: CREATE_NEW_PROCESS_GROUP so it doesn't die with our console.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            logger.info("ComfyUI started with pid %s", self._process.pid)
            return True
        except Exception:
            logger.exception("Failed to start ComfyUI")
            self._process = None
            return False

    def _wait_until_healthy(self) -> bool:
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            # Check if process died during startup
            if self._process is not None and self._process.poll() is not None:
                logger.error("ComfyUI process died during startup (exit code %s)", self._process.returncode)
                self._process = None
                return False

            if self._is_healthy():
                logger.info("ComfyUI is ready")
                return True

            time.sleep(_HEALTH_POLL_INTERVAL)

        logger.error("ComfyUI did not become healthy within %ss", _STARTUP_TIMEOUT)
        return False


# Module-level singleton
_instance: ComfyUIProcess | None = None


def get_comfyui_process() -> ComfyUIProcess:
    """Get or create the global ComfyUI process manager."""
    global _instance
    if _instance is None:
        _instance = ComfyUIProcess()
    return _instance
