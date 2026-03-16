"""Workflow template loading and parameter injection for ComfyUI proxy.

Each function loads a base workflow JSON from _services/workflows/,
injects the caller-supplied parameters, and returns a ready-to-queue dict.

The workflow JSONs are copies of the originals from gen.* directories
with any fixes applied (e.g. missing nodes). The originals are never modified.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Workflow templates live alongside this module in _services/workflows/
# ---------------------------------------------------------------------------
_WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"

_VIDEO_WORKFLOW = _WORKFLOWS_DIR / "video_animate.json"
_AV_WORKFLOW = _WORKFLOWS_DIR / "av_movie.json"
_IMAGE_SCENE_WORKFLOW = _WORKFLOWS_DIR / "image_scene.json"
_IMAGE_CHAR_WORKFLOW = _WORKFLOWS_DIR / "image_character.json"
_RETAKE_WORKFLOW = _WORKFLOWS_DIR / "retake_inpaint.json"


def _load(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _ensure_blank_image() -> str:
    """Create a blank black PNG in ComfyUI's input/ dir for T2V mode.

    Returns the filename (not full path) for use in LoadImage nodes.
    """
    comfyui_root = _WORKFLOWS_DIR.parent.parent.parent.parent / "ComfyUI"
    input_dir = comfyui_root / "input"
    blank_path = input_dir / "_blank_placeholder.png"
    if not blank_path.exists():
        from PIL import Image
        input_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (512, 512), (0, 0, 0))
        img.save(str(blank_path))
    return "_blank_placeholder.png"


# ===================================================================
# Video generation – uses movie.json (same as AV) with audio disabled
# ===================================================================

def build_video_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    image_filename: str | None = None,
    width: int = 1024,
    height: int = 576,
    frame_count: int = 73,
    fps: int = 24,
    seed: int | None = None,
    output_prefix: str = "ltxd_video",
) -> dict[str, Any]:
    """Build a video-only workflow from the movie.json template (audio disabled).

    This reuses the AV workflow with use_audio=False so it runs
    on the same LTX-2 19B pipeline with DualCLIP (Gemma 3).
    """
    return build_av_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_filename=image_filename,
        audio_filename=None,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=float(fps),
        seed=seed,
        use_audio=False,
        output_prefix=output_prefix,
    )


# ===================================================================
# Audio-Video generation (gen.av/workflow/movie.json)
# ===================================================================

def build_av_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    image_filename: str | None = None,
    audio_filename: str | None = None,
    width: int = 1024,
    height: int = 576,
    frame_count: int = 73,
    fps: float = 24.0,
    seed: int | None = None,
    use_lora_depth: bool = True,
    use_lora_canny: bool = True,
    use_lora_pose: bool = True,
    use_lora_detailer: bool = True,
    use_audio: bool = True,
    output_prefix: str = "ltxd_av",
) -> dict[str, Any]:
    """Build an LTX-2 audio-video workflow from movie.json template.

    Node mapping (movie.json):
      204         -> PrimitiveStringMultiline (positive prompt)
      282         -> CLIPTextEncode (negative prompt)
      228         -> LoadImage (guide image)
      281         -> LoadAudio (audio file)
      279:240     -> RandomNoise (seed - upscale pass)
      279:243     -> RandomNoise (seed - base pass)
      279:270     -> PrimitiveFloat (frame rate)
      279:277     -> PrimitiveInt (length / frame count)
      279:303     -> CM_IntToFloat (length for duration calc)
      279:281     -> LoRA depth (switch via 279:286)
      279:284     -> LoRA canny (switch via 279:288)
      279:283     -> LoRA pose (switch via 279:289)
      279:282     -> LoRA detailer (switch via 279:290)
      279:298     -> Switch (audio latent vs empty)
      229         -> ResizeImageMaskNode (image dimensions)
      104         -> SaveVideo (output)
    """
    wf = _load(_AV_WORKFLOW)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Prompts
    wf["204"]["inputs"]["value"] = prompt
    wf["282"]["inputs"]["text"] = negative_prompt

    # Guide image — LoadImage (228) must always reference a valid file.
    # For T2V (no image), create a blank placeholder so the node validates.
    if image_filename:
        wf["228"]["inputs"]["image"] = image_filename
    else:
        wf["228"]["inputs"]["image"] = _ensure_blank_image()
    wf["229"]["inputs"]["resize_type.width"] = width
    wf["229"]["inputs"]["resize_type.height"] = height

    # Audio — LoadAudio (281) must always reference a valid file even when
    # the audio path is switched off, because ComfyUI validates all nodes.
    if audio_filename and use_audio:
        wf["281"]["inputs"]["audio"] = audio_filename
        wf["279:298"]["inputs"]["switch"] = True
    else:
        # Point to a known placeholder; the switch keeps it unused.
        wf["281"]["inputs"]["audio"] = audio_filename or "narrator.wav"
        wf["279:298"]["inputs"]["switch"] = False

    # Frame rate and length
    wf["279:270"]["inputs"]["value"] = fps
    wf["279:277"]["inputs"]["value"] = frame_count
    wf["279:303"]["inputs"]["a"] = frame_count
    wf["279:304"]["inputs"]["b"] = fps  # Duration = frames / fps

    # Seeds
    wf["279:240"]["inputs"]["noise_seed"] = seed
    wf["279:243"]["inputs"]["noise_seed"] = seed + 1

    # LoRA switches
    wf["279:286"]["inputs"]["switch"] = use_lora_depth
    wf["279:288"]["inputs"]["switch"] = use_lora_canny
    wf["279:289"]["inputs"]["switch"] = use_lora_pose
    wf["279:290"]["inputs"]["switch"] = use_lora_detailer

    # LoRA overall enable
    wf["279:291"]["inputs"]["switch"] = any([use_lora_depth, use_lora_canny, use_lora_pose, use_lora_detailer])
    wf["279:292"]["inputs"]["switch"] = any([use_lora_depth, use_lora_canny, use_lora_pose, use_lora_detailer])

    # Output
    wf["104"]["inputs"]["filename_prefix"] = output_prefix

    return wf


# ===================================================================
# Image generation – Scene (gen.image/workflow/scene.json, Flux)
# ===================================================================

def build_image_scene_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1920,
    height: int = 1080,
    steps: int = 25,
    cfg: float = 1.0,
    guidance: float = 3.6,
    seed: int | None = None,
    scale_by: float = 1.0,
    output_prefix: str = "ltxd_scene",
) -> dict[str, Any]:
    """Build a Flux scene image generation workflow from scene.json template.

    Node mapping (scene.json):
      10 -> DualCLIPLoader (Flux CLIP)
      11 -> VAELoader (Flux VAE)
      16 -> KSampler (seed, steps, cfg, sampler, scheduler, denoise)
      19 -> EmptySD3LatentImage (width, height)
      20 -> VAEDecode
      21 -> SaveImage (filename_prefix)
      30 -> CLIPTextEncode (positive) [referenced by FluxGuidance]
      32 -> FluxGuidance (guidance scale)
      33 -> CLIPTextEncode (prompt text - feeds into negative via ConditioningZeroOut)
      34 -> ConditioningZeroOut (negative conditioning)
      41 -> UnetLoaderGGUF (Flux model)
      42 -> ImageScaleBy (scale_by)
    """
    wf = _load(_IMAGE_SCENE_WORKFLOW)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Node 30 is positive CLIPTextEncode -> FluxGuidance -> KSampler positive
    # Node 33 is the CLIPTextEncode that feeds negative (via ConditioningZeroOut)
    wf["30"]["inputs"]["text"] = prompt
    wf["33"]["inputs"]["text"] = prompt

    # FluxGuidance
    wf["32"]["inputs"]["guidance"] = guidance

    # KSampler
    wf["16"]["inputs"]["seed"] = seed
    wf["16"]["inputs"]["steps"] = steps
    wf["16"]["inputs"]["cfg"] = cfg

    # Latent dimensions
    wf["19"]["inputs"]["width"] = width
    wf["19"]["inputs"]["height"] = height

    # Output
    wf["21"]["inputs"]["filename_prefix"] = output_prefix

    # Scale
    wf["42"]["inputs"]["scale_by"] = scale_by

    return wf


# ===================================================================
# Image generation – Character (gen.image/workflow/character_location.json)
# ===================================================================

def build_image_character_workflow(
    *,
    prompt: str,
    width: int = 1024,
    height: int = 576,
    steps: int = 25,
    cfg: float = 1.0,
    seed: int | None = None,
    output_prefix: str = "ltxd_char",
) -> dict[str, Any]:
    """Build a Flux character/location image workflow from character_location.json.

    Node mapping (character_location.json):
      10 -> DualCLIPLoader (Flux CLIP)
      11 -> VAELoader (Flux VAE)
      16 -> KSampler (seed, steps, cfg)
      19 -> EmptySD3LatentImage (width, height)
      21 -> SaveImage (filename_prefix)
      33 -> CLIPTextEncode (prompt text)
      41 -> UnetLoaderGGUF (Flux model)
    """
    wf = _load(_IMAGE_CHAR_WORKFLOW)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    wf["33"]["inputs"]["text"] = prompt
    wf["16"]["inputs"]["seed"] = seed
    wf["16"]["inputs"]["steps"] = steps
    wf["16"]["inputs"]["cfg"] = cfg
    wf["19"]["inputs"]["width"] = width
    wf["19"]["inputs"]["height"] = height
    wf["21"]["inputs"]["filename_prefix"] = output_prefix

    return wf


# ===================================================================
# Retake / Temporal Inpainting (retake_inpaint.json)
# ===================================================================

def build_retake_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    video_filename: str,
    start_time: float = 0.5,
    end_time: float = 0.5,
    video_frames: int = 4,
    fps: int = 24,
    seed: int | None = None,
    output_prefix: str = "ltxd_retake",
) -> dict[str, Any]:
    """Build a temporal inpainting workflow from retake_inpaint.json.

    The LTXVAudioVideoMask node creates a mask that keeps start_time seconds
    at the beginning and end_time seconds at the end, regenerating the middle.

    Node mapping:
      10  -> CLIPTextEncode (positive prompt)
      11  -> CLIPTextEncode (negative prompt)
      20  -> VHS_LoadVideo (video file, force_rate)
      30  -> LTXVAudioVideoMask (video_start_time, video_end_time, etc.)
      43  -> RandomNoise (seed)
      70  -> SaveVideo (output prefix)
      80  -> SaveImage (frame output prefix)
    """
    wf = _load(_RETAKE_WORKFLOW)

    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Prompts
    wf["10"]["inputs"]["text"] = prompt
    wf["11"]["inputs"]["text"] = negative_prompt

    # Input video
    wf["20"]["inputs"]["video"] = video_filename
    wf["20"]["inputs"]["force_rate"] = fps

    # Temporal mask — start/end times define the region to regenerate
    wf["30"]["inputs"]["video_fps"] = float(fps)
    wf["30"]["inputs"]["video_start_time"] = start_time
    wf["30"]["inputs"]["video_end_time"] = start_time + end_time
    wf["30"]["inputs"]["audio_start_time"] = start_time
    wf["30"]["inputs"]["audio_end_time"] = start_time + end_time

    # Conditioning frame rate
    wf["40"]["inputs"]["frame_rate"] = float(fps)

    # Seed
    wf["43"]["inputs"]["noise_seed"] = seed

    # Output
    wf["70"]["inputs"]["filename_prefix"] = output_prefix
    wf["80"]["inputs"]["filename_prefix"] = f"{output_prefix}_frame"

    return wf


# ===================================================================
# Resolution helpers
# ===================================================================

_RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "360p": (640, 360),
    "480p": (854, 480),
    "512p": (910, 512),
    "540p": (960, 540),
    "576p": (1024, 576),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def resolve_resolution(
    resolution: str,
    aspect_ratio: str = "16:9",
) -> tuple[int, int]:
    """Convert a resolution string like '720p' to (width, height).

    Flips for 9:16 portrait.
    """
    w, h = _RESOLUTION_MAP.get(resolution, (1024, 576))
    if aspect_ratio == "9:16":
        w, h = h, w
    return w, h


def compute_frame_count(duration_seconds: float, fps: int = 24) -> int:
    """Compute frame count rounded to 8k+1 pattern required by LTXV."""
    n = int(duration_seconds * fps)
    n = ((n // 8) * 8) + 1
    return max(n, 9)  # minimum 9 frames
