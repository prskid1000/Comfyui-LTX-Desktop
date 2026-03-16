# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an Electron app for AI video generation using LTX models, with a **ComfyUI proxy layer** that routes all generation through local ComfyUI workflows. Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export
- **Backend** (`backend/`): Python FastAPI server (random port) handling ML model orchestration and generation
- **ComfyUI Proxy** (`backend/_services/` + `backend/_routes/comfyui_proxy.py`): Translates LTX Desktop API calls into ComfyUI workflow executions

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only |
| `pnpm backend:test` | Run Python pytest tests |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build` | Full platform build (auto-detects platform) |
| `pnpm setup:dev` | One-time dev environment setup (auto-detects platform) |

Run a single backend test file via pnpm: `pnpm backend:test -- tests/test_ic_lora.py`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## ComfyUI Integration Architecture

The ComfyUI proxy is toggle-based via `comfyuiEnabled` setting. When ON, all generation routes through ComfyUI. When OFF, native LTX pipelines are used.

### Data Flow (ComfyUI mode)

```
Frontend hook (use-generation/use-retake/use-ic-lora)
  → checks appSettings.comfyuiEnabled
  → POST /api/comfyui/generate/* (via backendFetch)
  → comfyui_proxy.py: _ensure_comfyui() auto-starts ComfyUI if needed
  → comfyui_workflows.py: loads JSON template, injects params
  → comfyui_client.py: POST /prompt to ComfyUI, polls /history/{id}
  → copies output files to LTX Desktop outputs dir
  → returns output_paths[] to frontend
```

### Required Directory Layout

ComfyUI and LTX-Desktop **must be siblings** under the same parent directory:

```
<root>/              # e.g. D:\.comfyui\
├── ComfyUI/         # ComfyUI installation
├── .venv/           # Python venv with ComfyUI dependencies
└── LTX-Desktop/     # This repo (backend/_services/ resolves paths via .parent x4)
```

The backend resolves `<root>` by walking up from `backend/_services/*.py` → `_services` → `backend` → `LTX-Desktop` → `<root>`. Then looks for `<root>/ComfyUI/` and `<root>/.venv/`.

**Environment variable overrides** (for non-standard layouts):
- `COMFYUI_DIR` — path to ComfyUI installation (overrides `<root>/ComfyUI`)
- `COMFYUI_BASE_URL` — ComfyUI HTTP URL (default `http://127.0.0.1:8188`)
- `COMFYUI_VENV_PYTHON` — Python executable with ComfyUI deps (overrides `<root>/.venv/Scripts/python.exe`)

### Key ComfyUI Files

| File | Purpose |
|---|---|
| `backend/_services/comfyui_client.py` | HTTP client wrapping ComfyUI `/prompt` and `/history` APIs |
| `backend/_services/comfyui_workflows.py` | Loads workflow JSON templates, injects parameters (prompt, seed, resolution, fps, etc.) |
| `backend/_services/comfyui_process.py` | Auto-starts ComfyUI subprocess in background thread, health polling |
| `backend/_services/comfyui_model_manager.py` | Scans ComfyUI model dirs, detects missing models, downloads from HuggingFace |
| `backend/_services/workflows/*.json` | Workflow templates (copies from gen.*, never modify originals) |
| `backend/_routes/comfyui_proxy.py` | All `/api/comfyui/*` FastAPI endpoints + progress tracking |
| `backend/state/app_settings.py` | `comfyui_enabled` and `comfyui_url` settings fields |

### Workflow Templates

Located in `backend/_services/workflows/`:

| File | Used For | Base Model |
|---|---|---|
| `av_movie.json` | Video (T2V/I2V), Audio-Video (A2V), IC-LoRA | LTX-2 19B GGUF + DualCLIP (Gemma 3) |
| `image_scene.json` | Scene image generation | Flux Kontext GGUF (Q8_0) |
| `image_character.json` | Character image generation | Flux Kontext GGUF (Q8_0) |
| `retake_inpaint.json` | Temporal inpainting (retake) | LTX-2 19B GGUF + distilled LoRA |

**Important**: These are copies of the gen.* originals with fixes applied. Never modify the original workflow JSONs in gen.* directories — always create/edit copies in `_services/workflows/`.

### ComfyUI API Endpoints

All registered under `/api/comfyui/` prefix in `comfyui_proxy.py`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check ComfyUI availability, trigger background start |
| `/progress` | GET | Poll generation progress (status, phase, progress%) |
| `/models/status` | GET | List all 16 required models with found/missing status |
| `/models/download` | POST | Download all missing models from HuggingFace |
| `/generate/video` | POST | Video generation (T2V/I2V via av_movie.json) |
| `/generate/av` | POST | Audio-video generation (A2V via av_movie.json + audio) |
| `/generate/image` | POST | Image generation (Flux via image_scene/character.json) |
| `/generate/raw` | POST | Submit arbitrary workflow JSON |
| `/retake` | POST | Temporal inpainting via retake_inpaint.json |
| `/ic-lora/generate` | POST | IC-LoRA conditioned video (depth/canny/pose + detailer) |

### Frontend ComfyUI Branching

Every generation hook checks `appSettings.comfyuiEnabled` to decide the route:

| Hook | Native Endpoint | ComfyUI Endpoint |
|---|---|---|
| `use-generation.ts` generate() | `/api/generate` | `/api/comfyui/generate/video` or `/generate/av` |
| `use-generation.ts` generateImage() | `/api/generate-image` | `/api/comfyui/generate/image` |
| `use-retake.ts` submitRetake() | `/api/retake` | `/api/comfyui/retake` |
| `use-ic-lora.ts` submitIcLora() | `/api/ic-lora/generate` | `/api/comfyui/ic-lora/generate` |

### Settings Behavior (ComfyUI ON vs OFF)

When `comfyuiEnabled` is true, the frontend hides irrelevant native options:

**Hidden in Settings Modal**: API Keys tab, Inference tab, Prompt Enhancer tab, text encoding, torch compile, model preload, seed lock, analytics

**Hidden in SettingsPanel**: Camera motion, audio on/off toggle (audio auto-detected from input)

**Hidden in Playground**: ModelStatusDropdown

**Changed in SettingsPanel/GenSpace PromptBar**: Model shows "LTX-2 19B (ComfyUI)", resolutions 360p-1080p, durations 1-10s, FPS 6/12/18/24/30

**Bypassed**: `sanitizeForcedApiVideoSettings()` — user selections stick without API constraints overriding them. `forceApiGenerations` gates are bypassed (IC-LoRA, retake, model download checks).

### IC-LoRA Panel (ComfyUI mode)

- Model download gate bypassed (`icLoraReady = true` when comfyuiEnabled)
- Conditioning extraction skipped (ComfyUI handles it internally)
- Conditioning column hidden, Output column hidden (result shows in main VideoPlayer)
- Only Input column visible (full width for driving video)
- Detailer toggle button in panel header (ON by default)
- Pose added as third conditioning type (depth/canny/pose)

### ComfyUI Process Manager

`comfyui_process.py` manages the ComfyUI subprocess:

- Auto-starts in a background thread (non-blocking)
- Uses `D:\.comfyui\.venv\Scripts\python.exe` with flags: `--listen --async-offload 16 --cache-none --disable-smart-memory --reserve-vram 0.3`
- Health polling every 3s during startup, 120s timeout
- `ensure_running()` — non-blocking, returns immediately
- `ensure_running_blocking()` — blocks until ready (used by generation endpoints)
- `CREATE_NEW_PROCESS_GROUP` on Windows so it survives console close

### Model Manager

`comfyui_model_manager.py` tracks 16 required models across 5 HuggingFace repos:

- Queries ComfyUI's folder structure (API or fallback to standard paths)
- `check_models()` — returns found/missing status for each model
- `download_missing_models()` — downloads all missing via `huggingface_hub`
- Files land directly in the correct ComfyUI model subdirectory

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*`
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`, `playground`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`)
- **Backend calls**: Always use `backendFetch` from `frontend/lib/backend.ts` for app backend HTTP requests (it attaches auth/session details). Do not call `fetch` directly for backend endpoints.
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- **No frontend tests** currently exist

## Backend Architecture

Request flow: `_routes/* (thin) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **ComfyUI services** (`_services/`): Separate from native services. Direct HTTP calls to ComfyUI, no AppHandler dependency. Registered via `comfyui_proxy_router` in `app_factory.py`.
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock→read/validate→unlock→heavy work→lock→write. Never hold lock during heavy compute/IO.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode is also enforced as a test (`test_pyright.py`)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

### Adding a ComfyUI Feature

1. Create or copy a workflow JSON into `backend/_services/workflows/` (never modify gen.* originals)
2. Add a builder function in `comfyui_workflows.py` that loads the template and injects parameters
3. Add request/response models and endpoint in `comfyui_proxy.py`
4. Add frontend branching in the relevant hook (`use-generation.ts`, `use-retake.ts`, `use-ic-lora.ts`)
5. Pass `comfyuiEnabled` to any UI component that needs to hide/show options

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS

## Python Config

- Python 3.13+ (per `.python-version`), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`)
- Dependencies in `backend/pyproject.toml`
- Backend has its own venv at `backend/.venv` (created by `uv sync`)
- ComfyUI uses a separate shared venv at `D:\.comfyui\.venv`

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`
- ComfyUI proxy routes: `backend/_routes/comfyui_proxy.py`
- ComfyUI workflow templates: `backend/_services/workflows/`
- ComfyUI process manager: `backend/_services/comfyui_process.py`
- ComfyUI model manager: `backend/_services/comfyui_model_manager.py`
- App settings (with comfyui fields): `backend/state/app_settings.py`
- Frontend settings context: `frontend/contexts/AppSettingsContext.tsx`
