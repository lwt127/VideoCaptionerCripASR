# VideoCaptioner + CrispASR (Integrated)

A desktop app for automatic video subtitling / captioning. This repository is the
**VideoCaptioner** application (based on
[WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner), v1.4.2),
integrated with **CrispASR** for local speech recognition / transcription.

Published so others can clone the project and run VideoCaptioner with CrispASR.

---

## Repository layout

```
main.py              # Application entry point  ->  python main.py
requirements.txt     # Python dependencies
app/                 # VideoCaptioner source (PyQt5 GUI, core, threads, views)
resource/            # Assets, subtitle styles, translations, and bundled binaries
  └─ bin/CrispASR/   # Prebuilt CrispASR binaries (Windows: crispasr.exe, *.dll)
_build/CrispASR/     # CrispASR engine SOURCE (build it for non-Windows platforms)
```

## What is NOT included (by design)

To keep the repository lean and avoid committing large binaries / private data,
the following are intentionally **not** tracked (see `.gitignore`):

- `runtime/` — the bundled portable Python runtime (recreate it yourself, see below)
- `VideoCaptioner/`, `AppData/` — runtime data, caches, logs, downloaded models
- `work-dir/` — local working media files
- CrispASR `build/`, `build-cuda/`, `models/` and other build artifacts
- Installer / frozen-app files (`*.exe` installers, `unins000.*`, backup zips)

> Because of this, `AppData/` (settings, logs, models, cache) and `work-dir/` are
> created at runtime. The app recreates the folders it needs on first launch.

---

## Quick start (run from source)

### 1. Clone

```bash
git clone https://github.com/lwt127/VideoCaptionerCripASR.git
cd VideoCaptionerCripASR
```

### 2. Create a Python environment (Python 3.11 recommended)

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Make sure `ffmpeg` is available

The app needs `ffmpeg` for audio/video processing. Install it system-wide (so it's
on your `PATH`) or place the binaries in `resource/bin/`.

### 4. Run

```bash
python main.py
```

---

## Transcription backends

VideoCaptioner can transcribe locally using several backends.

### CrispASR (integrated, zero-setup ✨)

CrispASR works **out of the box** — no manual installation. In the **语音转录
(Transcription)** tab choose **CrispASR**, pick a backend engine + model, then start
the task. On first use:

- If the **CrispASR engine** binary is missing, it is **automatically downloaded**
  from the official GitHub Releases (`CrispStrobe/CrispASR`) into
  `resource/bin/CrispASR/`.
- The selected **model** is **automatically downloaded** by CrispASR on first run
  (cached under `~/.cache/crispasr/`).

So users only need to *select* a backend/model and press start — everything else is
fetched automatically.

> Windows ships a self-contained prebuilt `crispasr.exe` (auto-downloaded if absent).
> To build manually for other platforms, see `_build/CrispASR/` (`build-windows.bat`,
> CMake presets) and copy the binary into `resource/bin/CrispASR/`.

**Available CrispASR backends** (all auto-downloadable): Whisper, Parakeet (TDT/RNNT/CTC
variants), FastConformer-CTC, Canary, Voxtral Mini 3B / 4B Realtime, Granite Speech,
Qwen3-ASR, Mega-ASR, Fun-ASR Nano, SenseVoice, Paraformer-zh, Cohere, wav2vec2 /
HuBERT / data2vec, omniASR (1600+ languages), FireRedASR2, GLM-ASR, Kyutai STT,
Gemma4-E2B, MiMo-ASR, MOSS-Audio, VibeVoice, KugelAudio, Moonshine — plus selectable
VAD methods (Silero / FireRedVAD / MarbleNet / Whisper-VAD).

### Faster-Whisper (optional)

Invoked as an **external program** (Faster-Whisper-XXL), not a Python package.
Download it and point the app to it via Settings, or place it under
`resource/bin/Faster-Whisper-XXL`.

### Models

CrispASR models download automatically (see above). WhisperCpp / Faster-Whisper
models can be downloaded via the in-app model manager into `AppData/models/`.

---

## (Optional) Recreate the portable runtime

The original distribution shipped a bundled `runtime/` (portable Python). That is
not committed here. To reproduce a fully portable build, recreate a Python 3.11
runtime and install `requirements.txt` into it, or just use the `.venv` setup above
for development.

---

## Components & licenses

- **VideoCaptioner** — see the upstream project:
  https://github.com/WEIFENG2333/VideoCaptioner
- **CrispASR** — see `_build/CrispASR/LICENSE`.

> Note: This is an integration/distribution repository assembled from a local
> install. Some setup steps (runtime, models, optional backends) must be performed
> locally because those large artifacts are not committed here.
