"""Async HTTP client for ComfyUI prompt API."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

COMFYUI_DEFAULT_URL = "http://127.0.0.1:8188"
POLL_INTERVAL_SECONDS = 2.0
SUBMIT_TIMEOUT_SECONDS = 120
POLL_TIMEOUT_SECONDS = 5


class ComfyUIClient:
    """Synchronous client that talks to a running ComfyUI instance."""

    def __init__(
        self,
        base_url: str = COMFYUI_DEFAULT_URL,
        comfyui_output_dir: str | None = None,
        comfyui_input_dir: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.comfyui_output_dir = comfyui_output_dir
        self.comfyui_input_dir = comfyui_input_dir

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if ComfyUI responds to a simple GET."""
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Prompt API
    # ------------------------------------------------------------------

    def queue_prompt(self, workflow: dict[str, Any], client_id: str | None = None) -> str:
        """POST /prompt – queue a workflow and return the prompt_id."""
        payload: dict[str, Any] = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        resp = requests.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=SUBMIT_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            # Log the full error from ComfyUI for debugging
            try:
                err_data = resp.json()
                node_errors = err_data.get("node_errors", {})
                if node_errors:
                    for nid, nerr in node_errors.items():
                        ct = nerr.get("class_type", "?")
                        for e in nerr.get("errors", []):
                            logger.error("ComfyUI node %s (%s): %s - %s",
                                         nid, ct, e.get("message", ""), e.get("details", ""))
            except Exception:
                pass
            resp.raise_for_status()
        prompt_id: str = resp.json()["prompt_id"]
        return prompt_id

    def poll_until_complete(
        self,
        prompt_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        on_progress: Any | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Poll GET /history/{prompt_id} until execution completes.

        ComfyUI history uses status.status_str ("success" | "error") and
        status.completed (bool). We check for both modern and legacy formats.

        Returns the full history entry for the prompt.
        Raises RuntimeError on execution errors, TimeoutError on deadline.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=POLL_TIMEOUT_SECONDS,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        status = entry.get("status", {})
                        status_str = status.get("status_str", "")

                        # Error: raise immediately
                        if status_str == "error":
                            err_msg = self._extract_error(status)
                            raise RuntimeError(f"ComfyUI execution error: {err_msg}")

                        # Success
                        if status.get("completed", False) or status_str == "success":
                            time.sleep(1)
                            return entry  # type: ignore[no-any-return]

                        # Legacy: check exec_info.queue_remaining
                        exec_info = status.get("exec_info", {})
                        if exec_info.get("queue_remaining", 1) == 0:
                            time.sleep(1)
                            return entry  # type: ignore[no-any-return]
            except requests.RequestException:
                pass

            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete within {timeout}s")

    @staticmethod
    def _extract_error(status: dict[str, Any]) -> str:
        """Pull the error message from a ComfyUI status dict."""
        for msg_type, msg_data in status.get("messages", []):
            if msg_type == "execution_error" and isinstance(msg_data, dict):
                return str(msg_data.get("exception_message", "unknown error"))
        return "unknown error"

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def copy_input_file(self, source_path: str, filename: str | None = None) -> str:
        """Copy a file into ComfyUI's input/ directory so workflows can reference it.

        Returns the filename (not full path) that workflows should use in LoadImage nodes.
        """
        if not self.comfyui_input_dir:
            raise RuntimeError("comfyui_input_dir not configured")
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        dest_name = filename or f"proxy_{uuid.uuid4().hex[:8]}_{src.name}"
        dest = Path(self.comfyui_input_dir) / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(src), str(dest))
        return dest_name

    def find_output_files(
        self,
        prefix: str,
        extensions: tuple[str, ...] = (".mp4", ".png", ".jpg", ".wav"),
        after_time: float | None = None,
    ) -> list[Path]:
        """Find output files in ComfyUI's output directory matching a prefix."""
        if not self.comfyui_output_dir:
            raise RuntimeError("comfyui_output_dir not configured")
        out_dir = Path(self.comfyui_output_dir)
        if not out_dir.exists():
            return []
        results: list[Path] = []
        for f in out_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in extensions:
                continue
            if prefix and not f.name.startswith(prefix):
                continue
            if after_time and f.stat().st_mtime < after_time:
                continue
            results.append(f)
        results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return results

    def get_output_from_history(self, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract output file info from a history entry's outputs dict."""
        outputs = history_entry.get("outputs", {})
        files: list[dict[str, Any]] = []
        for node_output in outputs.values():
            # SaveImage / VHS_VideoCombine / SaveVideo put results in images/videos/gifs
            for key in ("images", "videos", "gifs", "audio"):
                items = node_output.get(key, [])
                if isinstance(items, list):
                    files.extend(items)
        return files
