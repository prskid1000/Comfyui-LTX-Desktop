"""Auto-detect ComfyUI model directories and download missing models.

Scans ComfyUI's folder_paths config to find where each model type lives,
then checks which required models are present and downloads any that are missing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_COMFYUI_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")


# ===================================================================
# Required models for our workflows
# ===================================================================

@dataclass
class RequiredModel:
    """A model file required by our ComfyUI workflows."""
    filename: str
    folder_type: str  # ComfyUI folder type: unet, clip, vae, loras, latent_upscale_models, text_encoders
    repo_id: str
    repo_filename: str  # filename in the HF repo (may differ from local filename)
    size_hint: str  # human-readable size
    description: str


REQUIRED_MODELS: list[RequiredModel] = [
    # UNet / Diffusion models
    RequiredModel(
        filename="ltx-2-19b-distilled_Q4_K_M.gguf",
        folder_type="unet",
        repo_id="Kijai/LTXV2_comfy",
        repo_filename="diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
        size_hint="~10 GB",
        description="LTX-2 19B distilled video model (GGUF Q4)",
    ),
    RequiredModel(
        filename="flux1-kontext-dev-Q8_0.gguf",
        folder_type="unet",
        repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF",
        repo_filename="FLUX.1-Kontext-dev-Q8_0.gguf",
        size_hint="~12.7 GB",
        description="Flux Kontext image model (GGUF Q8)",
    ),
    # CLIP / Text Encoders
    RequiredModel(
        filename="t5xxl_fp8_e4m3fn.safetensors",
        folder_type="clip",
        repo_id="comfyanonymous/flux_text_encoders",
        repo_filename="t5xxl_fp8_e4m3fn.safetensors",
        size_hint="~4.9 GB",
        description="T5-XXL text encoder (FP8)",
    ),
    RequiredModel(
        filename="clip_l.safetensors",
        folder_type="clip",
        repo_id="comfyanonymous/flux_text_encoders",
        repo_filename="clip_l.safetensors",
        size_hint="~246 MB",
        description="CLIP-L text encoder",
    ),
    RequiredModel(
        filename="gemma_3_12B_it_fp8_e4m3fn.safetensors",
        folder_type="text_encoders",
        repo_id="Kijai/LTXV2_comfy",
        repo_filename="text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors",
        size_hint="~12 GB",
        description="Gemma 3 12B text encoder (FP8) for LTX-2",
    ),
    RequiredModel(
        filename="ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
        folder_type="clip",
        repo_id="Kijai/LTXV2_comfy",
        repo_filename="text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
        size_hint="~1.5 GB",
        description="LTX-2 embeddings connector for DualCLIP",
    ),
    # VAE
    RequiredModel(
        filename="LTX2_video_vae_bf16.safetensors",
        folder_type="vae",
        repo_id="Kijai/LTXV2_comfy",
        repo_filename="vae/LTX2_video_vae_bf16.safetensors",
        size_hint="~330 MB",
        description="LTX-2 video VAE (BF16)",
    ),
    RequiredModel(
        filename="LTX2_audio_vae_bf16.safetensors",
        folder_type="vae",
        repo_id="Kijai/LTXV2_comfy",
        repo_filename="vae/LTX2_audio_vae_bf16.safetensors",
        size_hint="~330 MB",
        description="LTX-2 audio VAE (BF16)",
    ),
    RequiredModel(
        filename="flux_kontext_vae.safetensors",
        folder_type="vae",
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_filename="vae/diffusion_pytorch_model.safetensors",
        size_hint="~335 MB",
        description="Flux VAE for image generation",
    ),
    # LoRAs
    RequiredModel(
        filename="ltx-2-19b-ic-lora-depth-control.safetensors",
        folder_type="loras",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-19b-ic-lora-depth-control.safetensors",
        size_hint="~625 MB",
        description="IC-LoRA depth control",
    ),
    RequiredModel(
        filename="ltx-2-19b-ic-lora-canny-control.safetensors",
        folder_type="loras",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-19b-ic-lora-canny-control.safetensors",
        size_hint="~625 MB",
        description="IC-LoRA canny control",
    ),
    RequiredModel(
        filename="ltx-2-19b-ic-lora-pose-control.safetensors",
        folder_type="loras",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-19b-ic-lora-pose-control.safetensors",
        size_hint="~625 MB",
        description="IC-LoRA pose control",
    ),
    RequiredModel(
        filename="ltx-2-19b-ic-lora-detailer.safetensors",
        folder_type="loras",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-19b-ic-lora-detailer.safetensors",
        size_hint="~2.5 GB",
        description="IC-LoRA detailer",
    ),
    # Upscalers
    RequiredModel(
        filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        folder_type="latent_upscale_models",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        size_hint="~996 MB",
        description="LTX-2 spatial upscaler (2x)",
    ),
    RequiredModel(
        filename="ltx-2-temporal-upscaler-x2-1.0.safetensors",
        folder_type="latent_upscale_models",
        repo_id="Lightricks/LTX-2",
        repo_filename="ltx-2-temporal-upscaler-x2-1.0.safetensors",
        size_hint="~262 MB",
        description="LTX-2 temporal upscaler (2x frames)",
    ),
]


# ===================================================================
# Folder detection
# ===================================================================

def _get_comfyui_folder_paths(comfyui_url: str = _COMFYUI_URL) -> dict[str, list[str]]:
    """Query ComfyUI's /folder_paths endpoint to find model directories.

    Falls back to standard ComfyUI directory layout if the endpoint is unavailable.
    """
    # Try the API first (ComfyUI 0.14+)
    try:
        resp = requests.get(f"{comfyui_url}/folder_paths", timeout=5)
        if resp.status_code == 200:
            return resp.json()  # type: ignore[no-any-return]
    except Exception:
        pass

    # Fallback: derive from ComfyUI installation path (sibling of LTX-Desktop)
    comfyui_root = Path(__file__).resolve().parent.parent.parent.parent
    comfyui_dir = Path(os.environ.get("COMFYUI_DIR", str(comfyui_root / "ComfyUI")))
    models_dir = comfyui_dir / "models"

    return {
        "unet": [str(models_dir / "unet"), str(models_dir / "diffusion_models")],
        "clip": [str(models_dir / "clip")],
        "text_encoders": [str(models_dir / "text_encoders")],
        "vae": [str(models_dir / "vae")],
        "loras": [str(models_dir / "loras")],
        "latent_upscale_models": [str(models_dir / "latent_upscale_models")],
    }


def _find_model_file(filename: str, dirs: list[str]) -> Path | None:
    """Search for a model file across multiple directories."""
    for d in dirs:
        p = Path(d) / filename
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


# ===================================================================
# Status check
# ===================================================================

@dataclass
class ModelStatus:
    model: RequiredModel
    found: bool
    path: str | None


def check_models(comfyui_url: str = _COMFYUI_URL) -> list[ModelStatus]:
    """Check which required models are present and which are missing."""
    folder_paths = _get_comfyui_folder_paths(comfyui_url)
    results: list[ModelStatus] = []

    for model in REQUIRED_MODELS:
        dirs = folder_paths.get(model.folder_type, [])
        found_path = _find_model_file(model.filename, dirs)
        results.append(ModelStatus(
            model=model,
            found=found_path is not None,
            path=str(found_path) if found_path else None,
        ))

    return results


def get_missing_models(comfyui_url: str = _COMFYUI_URL) -> list[RequiredModel]:
    """Return only the models that are missing."""
    statuses = check_models(comfyui_url)
    return [s.model for s in statuses if not s.found]


# ===================================================================
# Download
# ===================================================================

def download_model(
    model: RequiredModel,
    comfyui_url: str = _COMFYUI_URL,
) -> str:
    """Download a single model to the correct ComfyUI directory.

    Returns the path where the model was saved.
    """
    from huggingface_hub import hf_hub_download

    folder_paths = _get_comfyui_folder_paths(comfyui_url)
    dirs = folder_paths.get(model.folder_type, [])
    if not dirs:
        raise RuntimeError(f"No directory found for folder type: {model.folder_type}")

    # Use the first directory
    target_dir = dirs[0]
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s (%s) from %s to %s",
                model.filename, model.size_hint, model.repo_id, target_dir)

    path = hf_hub_download(
        repo_id=model.repo_id,
        filename=model.repo_filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    # If the downloaded file is in a subdirectory, move it to the target dir root
    downloaded = Path(path)
    target_file = Path(target_dir) / model.filename
    if downloaded != target_file and downloaded.exists():
        import shutil
        shutil.move(str(downloaded), str(target_file))
        # Clean up empty subdirectories
        for parent in downloaded.parents:
            if parent == Path(target_dir):
                break
            try:
                parent.rmdir()
            except OSError:
                break

    final_path = str(target_file) if target_file.exists() else str(downloaded)
    logger.info("Downloaded: %s", final_path)
    return final_path


def download_missing_models(
    comfyui_url: str = _COMFYUI_URL,
    on_model_start: Any | None = None,
    on_model_done: Any | None = None,
) -> list[str]:
    """Download all missing models. Returns list of downloaded file paths."""
    missing = get_missing_models(comfyui_url)
    if not missing:
        logger.info("All required models are present")
        return []

    logger.info("Missing %d models: %s", len(missing), [m.filename for m in missing])
    downloaded: list[str] = []

    for model in missing:
        if on_model_start:
            on_model_start(model)
        try:
            path = download_model(model, comfyui_url)
            downloaded.append(path)
            if on_model_done:
                on_model_done(model, path, None)
        except Exception as exc:
            logger.error("Failed to download %s: %s", model.filename, exc)
            if on_model_done:
                on_model_done(model, None, str(exc))

    return downloaded
