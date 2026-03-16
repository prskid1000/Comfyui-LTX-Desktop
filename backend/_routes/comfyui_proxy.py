"""ComfyUI proxy routes – translate LTX Desktop API requests into ComfyUI workflow executions.

Supports all generation modes via existing gen.* workflows running on a local ComfyUI instance.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from _routes._errors import HTTPError
from _services.comfyui_client import ComfyUIClient
from _services.comfyui_model_manager import check_models, download_missing_models, get_missing_models
from _services.comfyui_process import get_comfyui_process
from _services.comfyui_workflows import (
    build_av_workflow,
    build_image_character_workflow,
    build_image_scene_workflow,
    build_retake_workflow,
    build_video_workflow,
    compute_frame_count,
    resolve_resolution,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/comfyui", tags=["comfyui-proxy"])

# ---------------------------------------------------------------------------
# Configuration (from environment or defaults)
# ---------------------------------------------------------------------------
_COMFYUI_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
_COMFYUI_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # D:\.comfyui
_COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", str(_COMFYUI_ROOT / "ComfyUI")))
_COMFYUI_INPUT = _COMFYUI_DIR / "input"
_COMFYUI_OUTPUT = _COMFYUI_DIR / "output"
_PROXY_OUTPUT = _COMFYUI_ROOT / "LTX-Desktop" / "outputs"

# Simple progress state for ComfyUI generations
_comfyui_generation_state: dict[str, Any] = {
    "status": "idle",
    "phase": "",
    "progress": 0,
    "prompt_id": None,
}
_state_lock = threading.Lock()


def _update_progress(status: str, phase: str, progress: int, prompt_id: str | None = None) -> None:
    with _state_lock:
        _comfyui_generation_state["status"] = status
        _comfyui_generation_state["phase"] = phase
        _comfyui_generation_state["progress"] = progress
        if prompt_id is not None:
            _comfyui_generation_state["prompt_id"] = prompt_id


def _get_comfyui_url() -> str:
    """Get ComfyUI URL from settings or environment."""
    try:
        from state import get_state_service
        handler = get_state_service()
        settings = handler.settings.get_settings_snapshot()
        if settings.comfyui_url:
            return settings.comfyui_url
    except Exception:
        pass
    return _COMFYUI_URL


def _get_client() -> ComfyUIClient:
    return ComfyUIClient(
        base_url=_get_comfyui_url(),
        comfyui_output_dir=str(_COMFYUI_OUTPUT),
        comfyui_input_dir=str(_COMFYUI_INPUT),
    )


def _ensure_comfyui() -> ComfyUIClient:
    """Get a client and auto-start ComfyUI if it's not running.

    Blocks until ComfyUI is ready (up to ~2 minutes).
    Raises HTTPError 503 if ComfyUI cannot be started.
    """
    url = _get_comfyui_url()
    proc = get_comfyui_process()

    if not proc.ensure_running_blocking(url=url, timeout=150):
        raise HTTPError(503, "ComfyUI is not available and could not be started")

    return _get_client()


def _ensure_output_dir() -> Path:
    _PROXY_OUTPUT.mkdir(parents=True, exist_ok=True)
    return _PROXY_OUTPUT


def _copy_to_outputs(src: Path, prefix: str, ext: str) -> str:
    """Copy a ComfyUI output file to the LTX Desktop outputs directory."""
    out_dir = _ensure_output_dir()
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]
    dest = out_dir / f"{prefix}_{ts}_{uid}{ext}"
    shutil.copy2(str(src), str(dest))
    return str(dest)


# ===================================================================
# Request / Response models
# ===================================================================


class ComfyUIVideoRequest(BaseModel):
    """Generate video via ComfyUI using gen.video/workflow/animate.json."""
    prompt: str
    negative_prompt: str = ""
    resolution: str = "576p"
    aspect_ratio: str = "16:9"
    duration: float = 4.0
    fps: int = 24
    image_path: str | None = None
    seed: int | None = None
    output_prefix: str = "ltxd_video"


class ComfyUIAVRequest(BaseModel):
    """Generate audio-video via ComfyUI using gen.av/workflow/movie.json."""
    prompt: str
    negative_prompt: str = ""
    resolution: str = "576p"
    aspect_ratio: str = "16:9"
    duration: float = 3.0
    fps: float = 24.0
    image_path: str | None = None
    audio_path: str | None = None
    seed: int | None = None
    use_lora_depth: bool = True
    use_lora_canny: bool = True
    use_lora_pose: bool = True
    use_lora_detailer: bool = True
    output_prefix: str = "ltxd_av"


class ComfyUIImageRequest(BaseModel):
    """Generate image via ComfyUI using gen.image Flux workflows."""
    prompt: str
    width: int = 1920
    height: int = 1080
    steps: int = 25
    cfg: float = 1.0
    guidance: float = 3.6
    seed: int | None = None
    scale_by: float = 1.0
    workflow_type: str = "scene"
    output_prefix: str = "ltxd_image"


class ComfyUIRetakeRequest(BaseModel):
    """Retake (re-generate) a section of video via ComfyUI."""
    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: str = "replace_audio_and_video"
    seed: int | None = None
    output_prefix: str = "ltxd_retake"


class ComfyUIIcLoraRequest(BaseModel):
    """Generate video with IC-LoRA conditioning via ComfyUI."""
    video_path: str
    conditioning_type: str = "depth"
    conditioning_strength: float = 1.0
    prompt: str
    seed: int | None = None
    output_prefix: str = "ltxd_iclora"


class ComfyUIRawWorkflowRequest(BaseModel):
    """Submit a raw ComfyUI workflow JSON directly."""
    workflow: dict[str, Any]
    output_prefix: str = "ltxd_raw"
    timeout: float = 1800


class ComfyUIGenerateResponse(BaseModel):
    status: str
    output_paths: list[str] = Field(default_factory=list)
    prompt_id: str | None = None
    comfyui_outputs: list[dict[str, Any]] = Field(default_factory=list)


class ComfyUIHealthResponse(BaseModel):
    available: bool
    url: str


class ComfyUIProgressResponse(BaseModel):
    status: str
    phase: str
    progress: int
    prompt_id: str | None = None


class ComfyUIModelStatusItem(BaseModel):
    filename: str
    folder_type: str
    found: bool
    path: str | None = None
    size_hint: str
    description: str


class ComfyUIModelsStatusResponse(BaseModel):
    total: int
    found: int
    missing: int
    models: list[ComfyUIModelStatusItem]


class ComfyUIModelsDownloadResponse(BaseModel):
    status: str
    downloaded: list[str] = Field(default_factory=list)
    missing_before: int = 0
    errors: list[str] = Field(default_factory=list)


# ===================================================================
# Endpoints
# ===================================================================


@router.get("/health", response_model=ComfyUIHealthResponse)
def comfyui_health() -> ComfyUIHealthResponse:
    """Check if the ComfyUI instance is reachable."""
    client = _get_client()
    url = _get_comfyui_url()
    proc = get_comfyui_process()
    available = client.is_available()

    # Kick off background start if not running (non-blocking)
    if not available:
        proc.ensure_running(url)

    return ComfyUIHealthResponse(
        available=available,
        url=url,
    )


@router.get("/progress", response_model=ComfyUIProgressResponse)
def comfyui_progress() -> ComfyUIProgressResponse:
    with _state_lock:
        return ComfyUIProgressResponse(**_comfyui_generation_state)


@router.get("/models/status", response_model=ComfyUIModelsStatusResponse)
def comfyui_models_status() -> ComfyUIModelsStatusResponse:
    """Check which required ComfyUI models are present and which are missing."""
    url = _get_comfyui_url()
    statuses = check_models(url)
    items = [
        ComfyUIModelStatusItem(
            filename=s.model.filename,
            folder_type=s.model.folder_type,
            found=s.found,
            path=s.path,
            size_hint=s.model.size_hint,
            description=s.model.description,
        )
        for s in statuses
    ]
    found_count = sum(1 for s in statuses if s.found)
    return ComfyUIModelsStatusResponse(
        total=len(statuses),
        found=found_count,
        missing=len(statuses) - found_count,
        models=items,
    )


@router.post("/models/download", response_model=ComfyUIModelsDownloadResponse)
def comfyui_models_download() -> ComfyUIModelsDownloadResponse:
    """Download all missing ComfyUI models from HuggingFace."""
    url = _get_comfyui_url()
    missing = get_missing_models(url)
    if not missing:
        return ComfyUIModelsDownloadResponse(status="complete", missing_before=0)

    errors: list[str] = []

    def on_done(model: Any, path: str | None, error: str | None) -> None:
        if error:
            errors.append(f"{model.filename}: {error}")

    downloaded = download_missing_models(url, on_model_done=on_done)
    return ComfyUIModelsDownloadResponse(
        status="complete" if not errors else "partial",
        downloaded=downloaded,
        missing_before=len(missing),
        errors=errors,
    )


@router.post("/generate/video", response_model=ComfyUIGenerateResponse)
def comfyui_generate_video(req: ComfyUIVideoRequest) -> ComfyUIGenerateResponse:
    """Generate video via ComfyUI LTXV workflow (gen.video/workflow/animate.json).

    Supports text-to-video and image-to-video.
    """
    client = _ensure_comfyui()

    w, h = resolve_resolution(req.resolution, req.aspect_ratio)
    frames = compute_frame_count(req.duration, req.fps)

    # Copy input image to ComfyUI if provided
    image_filename: str | None = None
    if req.image_path:
        image_filename = client.copy_input_file(req.image_path)

    workflow = build_video_workflow(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image_filename=image_filename,
        width=w,
        height=h,
        frame_count=frames,
        fps=req.fps,
        seed=req.seed,
        output_prefix=req.output_prefix,
    )

    return _execute_workflow(client, workflow, req.output_prefix, (".mp4",))


@router.post("/generate/av", response_model=ComfyUIGenerateResponse)
def comfyui_generate_av(req: ComfyUIAVRequest) -> ComfyUIGenerateResponse:
    """Generate audio-video via ComfyUI LTX-2 workflow (gen.av/workflow/movie.json).

    Supports image-to-video with audio synchronization.
    """
    client = _ensure_comfyui()

    w, h = resolve_resolution(req.resolution, req.aspect_ratio)
    frames = compute_frame_count(req.duration, int(req.fps))

    image_filename: str | None = None
    if req.image_path:
        image_filename = client.copy_input_file(req.image_path)

    audio_filename: str | None = None
    if req.audio_path:
        audio_filename = client.copy_input_file(req.audio_path)

    workflow = build_av_workflow(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image_filename=image_filename,
        audio_filename=audio_filename,
        width=w,
        height=h,
        frame_count=frames,
        fps=req.fps,
        seed=req.seed,
        use_lora_depth=req.use_lora_depth,
        use_lora_canny=req.use_lora_canny,
        use_lora_pose=req.use_lora_pose,
        use_lora_detailer=req.use_lora_detailer,
        use_audio=audio_filename is not None,
        output_prefix=req.output_prefix,
    )

    return _execute_workflow(client, workflow, req.output_prefix, (".mp4",))


@router.post("/generate/image", response_model=ComfyUIGenerateResponse)
def comfyui_generate_image(req: ComfyUIImageRequest) -> ComfyUIGenerateResponse:
    """Generate image via ComfyUI Flux workflow (gen.image/workflow/scene.json or character_location.json)."""
    client = _ensure_comfyui()

    if req.workflow_type == "character":
        workflow = build_image_character_workflow(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            output_prefix=req.output_prefix,
        )
    else:
        workflow = build_image_scene_workflow(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg=req.cfg,
            guidance=req.guidance,
            seed=req.seed,
            scale_by=req.scale_by,
            output_prefix=req.output_prefix,
        )

    return _execute_workflow(client, workflow, req.output_prefix, (".png", ".jpg"))


@router.post("/generate/raw", response_model=ComfyUIGenerateResponse)
def comfyui_generate_raw(req: ComfyUIRawWorkflowRequest) -> ComfyUIGenerateResponse:
    """Submit a raw ComfyUI workflow JSON directly.

    Use this for custom workflows not covered by the typed endpoints.
    """
    client = _ensure_comfyui()

    return _execute_workflow(
        client,
        req.workflow,
        req.output_prefix,
        (".mp4", ".png", ".jpg", ".wav"),
        timeout=req.timeout,
    )


@router.post("/retake", response_model=ComfyUIGenerateResponse)
def comfyui_retake(req: ComfyUIRetakeRequest) -> ComfyUIGenerateResponse:
    """Retake (re-generate) a video section via ComfyUI temporal inpainting.

    Uses LTXVAudioVideoMask to keep the beginning and end of the video
    intact while regenerating the middle section with a new prompt.
    """
    client = _ensure_comfyui()

    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPError(400, f"Video file not found: {req.video_path}")

    # Copy input video to ComfyUI input dir
    video_filename = client.copy_input_file(req.video_path)

    workflow = build_retake_workflow(
        prompt=req.prompt or "continuation of the scene",
        video_filename=video_filename,
        start_time=req.start_time,
        end_time=max(0.0, req.start_time),  # Keep symmetric padding
        video_frames=4,
        fps=24,
        seed=req.seed,
        output_prefix=req.output_prefix,
    )

    return _execute_workflow(client, workflow, req.output_prefix, (".mp4",))


@router.post("/ic-lora/generate", response_model=ComfyUIGenerateResponse)
def comfyui_ic_lora_generate(req: ComfyUIIcLoraRequest) -> ComfyUIGenerateResponse:
    """Generate video with IC-LoRA conditioning via ComfyUI.

    Uses movie.json workflow with specific LoRA switches enabled based
    on the conditioning type (depth, canny, or pose).
    """
    client = _ensure_comfyui()

    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPError(400, f"Video file not found: {req.video_path}")

    # Extract first frame as guide image
    frame_path = _extract_frame(req.video_path, 0)
    image_filename = client.copy_input_file(str(frame_path))

    # Map conditioning type to LoRA switches
    use_depth = req.conditioning_type == "depth"
    use_canny = req.conditioning_type == "canny"
    use_pose = req.conditioning_type == "pose"

    frames = compute_frame_count(3.0, 24)  # Default 3s for IC-LoRA
    workflow = build_av_workflow(
        prompt=req.prompt,
        image_filename=image_filename,
        width=768,
        height=432,
        frame_count=frames,
        fps=24.0,
        seed=req.seed,
        use_lora_depth=use_depth,
        use_lora_canny=use_canny,
        use_lora_pose=use_pose,
        use_lora_detailer=True,
        use_audio=False,
        output_prefix=req.output_prefix,
    )

    return _execute_workflow(client, workflow, req.output_prefix, (".mp4",))


# ===================================================================
# Internal helpers
# ===================================================================


def _extract_frame(video_path: str, time_seconds: float) -> Path:
    """Extract a single frame from a video at the given timestamp.

    Returns path to the extracted PNG frame in a temp location.
    """
    import subprocess as sp

    out_dir = _ensure_output_dir()
    uid = uuid.uuid4().hex[:8]
    frame_file = out_dir / f"_frame_{uid}.png"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_seconds),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        str(frame_file),
    ]
    try:
        sp.run(cmd, capture_output=True, timeout=30, check=True)
    except (sp.CalledProcessError, FileNotFoundError) as exc:
        raise HTTPError(500, f"Failed to extract frame: {exc}") from exc

    if not frame_file.exists():
        raise HTTPError(500, "Frame extraction produced no output")

    return frame_file


def _execute_workflow(
    client: ComfyUIClient,
    workflow: dict[str, Any],
    output_prefix: str,
    extensions: tuple[str, ...],
    timeout: float = 1800,
) -> ComfyUIGenerateResponse:
    """Queue a workflow, wait for completion, collect outputs."""
    before_time = time.time()

    _update_progress("running", "starting_comfyui", 2)

    # Auto-start ComfyUI if needed (blocks until ready)
    url = _get_comfyui_url()
    proc = get_comfyui_process()
    if not proc.ensure_running_blocking(url=url, timeout=150):
        _update_progress("error", "error", 0)
        raise HTTPError(503, "ComfyUI is not available and could not be started")

    _update_progress("running", "queuing", 5)

    try:
        prompt_id = client.queue_prompt(workflow)
    except Exception as exc:
        _update_progress("error", "error", 0)
        raise HTTPError(502, f"Failed to queue ComfyUI prompt: {exc}") from exc

    _update_progress("running", "waiting", 10, prompt_id=prompt_id)

    try:
        history = client.poll_until_complete(prompt_id, timeout=timeout)
    except TimeoutError as exc:
        _update_progress("error", "error", 0)
        raise HTTPError(504, str(exc)) from exc
    except Exception as exc:
        _update_progress("error", "error", 0)
        raise HTTPError(502, f"Error polling ComfyUI: {exc}") from exc

    _update_progress("running", "generating", 90, prompt_id=prompt_id)

    # Collect outputs from history
    comfyui_outputs = client.get_output_from_history(history)

    # Also scan filesystem for output files
    output_files = client.find_output_files(
        prefix=output_prefix,
        extensions=extensions,
        after_time=before_time,
    )

    # Copy outputs to LTX Desktop outputs directory
    output_paths: list[str] = []
    for f in output_files:
        dest = _copy_to_outputs(f, output_prefix, f.suffix)
        output_paths.append(dest)

    _update_progress("idle", "complete", 100, prompt_id=prompt_id)

    return ComfyUIGenerateResponse(
        status="complete",
        output_paths=output_paths,
        prompt_id=prompt_id,
        comfyui_outputs=comfyui_outputs,
    )
