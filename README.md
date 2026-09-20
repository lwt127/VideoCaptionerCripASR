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
- **GPU acceleration:** if an NVIDIA GPU (CUDA) is detected, the app automatically
  downloads/uses the **CUDA build** of the engine for much faster transcription;
  otherwise it uses the CPU build.
- The selected **model** is **automatically downloaded** by CrispASR on first run
  (cached under `C:\Users\<username>\.cache\crispasr\` on Windows).

So users only need to *select* a backend/model and press start — everything else
(engine, GPU build, model) is fetched automatically.

**Model licenses:** SenseVoice, Paraformer, Fun-ASR, Gemma and the German
Moonshine models require explicit confirmation of their respective terms.
Use **查看并确认授权** in the CrispASR settings, or review the dialog when starting
a single/batch task. Cancelling starts no transcription. Approval is saved for
the selected model and reviewed license revision, not for every model. German
Moonshine weights are non-commercial; other licenses have different conditions.
Queued batch tasks retain the model and approval selected when submitted.

Whisper and Kotoba selections download the exact requested filename when no
exact local model exists; they no longer silently fall back to Whisper Base.

> Windows ships a self-contained prebuilt `crispasr.exe` (auto-downloaded if absent).
> To build manually for other platforms, see `_build/CrispASR/` (`build-windows.bat`,
> CMake presets) and copy the binary into `resource/bin/CrispASR/`.

**Available CrispASR backends** (all auto-downloadable): Whisper, Parakeet (TDT/RNNT/CTC
variants), FastConformer-CTC, Canary, Voxtral Mini 3B / 4B Realtime, Granite Speech,
Qwen3-ASR, Mega-ASR, Fun-ASR Nano, SenseVoice, Paraformer-zh, Cohere, wav2vec2 /
HuBERT / data2vec, omniASR (1600+ languages), FireRedASR2, GLM-ASR, Kyutai STT,
Gemma4-E2B, MiMo-ASR, MOSS-Audio, VibeVoice, Moonshine — plus selectable
VAD methods (Silero / FireRedVAD / MarbleNet / Whisper-VAD).

### Interface languages

Settings supports Simplified Chinese, Traditional Chinese, English, and a
distinct Follow System option. Restart the application after changing the UI
language. Hong Kong/Macau locales use Traditional Chinese; unsupported system
languages fall back to English. Invalid saved locale values safely use Auto.

CrispASR model/GPU/alignment controls and license consent are translated using
`resource/translations/supplemental_ui.json`, loaded on top of the existing Qt
QM catalogs by `app/common/i18n.py`. This UTF-8 supplement needs no compilation;
new distributable builds must include the JSON alongside the existing QM files.

Backend, model, VAD and LID selections now store stable identifiers rather than
translated display text. Recognized old Chinese labels remain compatible and
are migrated when the settings panel opens; task snapshots are normalized too.
Unknown/removed selections are preserved and require reselection, never silently
substituted with another model. Interface language does not change ASR language,
translation target, license identity or subtitle layout values.

### CrispASR v0.8.34 model and timing controls

Additional choices include ReazonSpeech NeMo v2, Japanese Parakeet CTC 1.1B,
Qwen3 Japanese-anime, Cohere Arabic, GigaAM v3, Canary-Qwen, Higgs-STT,
MOSS-Transcribe, Kyutai STT 2.6B, and Voxtral Q8/F16 variants. Model variants
have separate language constraints (for example, Kyutai 2.6B and MOSS-Transcribe
are English-only). KugelAudio is TTS-only and is no longer an ASR choice.
Models with uncertain language/license metadata or placeholder downloads are
not automatically added merely because the engine lists a backend.

**时间戳对齐** controls alignment during transcription, not standalone retiming
of an existing subtitle file:

- **引擎默认** preserves existing behavior, including Canary's implicit aligner.
- **仅原生时间戳** disables implicit aligner downloads.
- **补充对齐** supplies an aligner without replacing existing word timings; it
  only runs on backends supporting CTC alignment.
- **强制重新对齐** asks the selected aligner to replace native timings.

Choose Qwen3, Canary CTC, English FastConformer, or a language-specific wav2vec2
aligner. These download separately on first use. Single-language aligners
require an explicit matching source language; Canary CTC is not a Japanese or
Chinese aligner. Canary/FastConformer weights require CC-BY attribution;
wav2vec2/Qwen3 use Apache-2.0 (retain their upstream notices).

**逐字时间戳输出** emits a cue per word (usually a character in CJK). It requires
real word data rather than silently interpolating timings; missing data fails
the task. Downstream requests for word timestamps use the same requirement.
Choose an appropriate aligner if the ASR backend cannot supply word timings.
Explicit timing output is not passed through the app's pause-extension heuristic.
Forced alignment is a request, not a guarantee: upstream can reject an invalid
alignment and retain native timing.

**运算引擎版本** offers Auto, CUDA 12, CUDA 13, Vulkan, CPU, and Legacy CPU.
Auto uses CUDA 12 when NVIDIA is detected, otherwise standard CPU. It does not
automatically certify CPU/GPU compatibility or select Legacy CPU.
CUDA 13 requires Turing or newer and driver 580+; GPU builds and standard CPU
require AVX2/FMA/F16C. Vulkan requires a compatible Vulkan 1.2 driver with
16-bit storage. Legacy CPU targets generic x86-64/SSE2. Full CUDA packages
bundle their runtime DLLs; a separate CUDA Toolkit is unnecessary.

Turning **GPU 加速** off selects a CPU-only engine so alignment also stays on
CPU. Each managed flavor installs into `resource/bin/CrispASR/<flavor>/`.
The downloader selects exact full release assets, checks package dependencies
and the version/backend/ABI banner, and stages replacements with rollback.
Those checks do not prove successful inference on a particular GPU/driver.

#### Cohere multilingual segments (experimental)

Choose **CrispASR → Cohere → Cohere Transcribe**, then enable
**Cohere 多语言分段模式（实验性）**. It is off by default and available only for
the Cohere backend. This mode overrides the source-language and LID selections
without changing their saved values.

- The selected CrispASR **VAD method** (Silero, FireRed, MarbleNet or Whisper-VAD)
  detects speech via `--vad-export-raw`, without loading an ASR model. There is
  no FFmpeg silence detection in this mode. VAD is required even if the saved
  normal-mode VAD switch is off; the method selector stays enabled.
- Exported speech regions get up to 150 ms padding. CrispASR may merge nearby
  regions or short utterances before export; these are not language boundaries.
  VAD runs in at most 30-minute batches to bound memory/sample counts. FFmpeg
  only extracts audio and ffprobe reads duration; batch edges may cut speech.
- `CRISPASR_VAD_FAILOVER=0` and `--require-vad` prevent the engine's normal
  full-clip fallback; reported failures abort rather than using silence detection.
- Windows are at most 20 seconds; uninterrupted speech requires hard cuts and
  words at those cuts can be affected. Long VAD regions are divided evenly to
  avoid creating subsecond tail windows with independent language probes.
  Short utterances are kept, not merged
  by the application, but their language detection is less reliable.
- Each window runs Cohere with `-l auto --lid-backend probe`, without running
  VAD a second time inside each window. The engine's fallback to Whisper-tiny
  remains in effect when self-probe is unavailable (e.g. an older GGUF without
  language metadata); the reason appears in the log.
- Local subtitle times are offset back to the original timeline, clamped to
  each window, and merged into one SRT. Silent windows produce no cues.
- **Slow:** the CLI reloads the model for each window and probes its supported
  languages. This first implementation does not keep a persistent model session.
- Use a multilingual Cohere model. A Japanese-only fine-tune does not gain new
  languages. Unsupported languages, within-sentence code-switching, overlapping
  speakers, and frequent language changes inside a window remain limitations.

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
