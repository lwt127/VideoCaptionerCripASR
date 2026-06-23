# VideoCaptioner + CrispASR (Integrated)

This repository contains the **VideoCaptioner** desktop application source code,
integrated with **CrispASR** for local speech recognition / transcription.

It is published so other users can pull the project and run VideoCaptioner with
CrispASR support.

## Repository layout

```
app/                 # VideoCaptioner application source (PyQt GUI, core, threads, views)
resource/            # Assets, subtitle styles, translations, bundled tools
_build/CrispASR/     # Integrated CrispASR engine (source only; build outputs ignored)
```

## What is NOT included (by design)

To keep the repository lean and avoid committing large binaries / private data,
the following are intentionally **not** tracked (see `.gitignore`):

- `runtime/` — the bundled portable Python runtime (rebuild/download separately)
- `VideoCaptioner/`, `AppData/` — local runtime data, caches, logs, downloaded models
- `work-dir/` — local working files (your own media)
- CrispASR `build/`, `build-cuda/`, `models/` and other build artifacts
- Installer/frozen-app files (`*.exe`, `unins000.*`, `_pystand_static.int`, backup zips)

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/lwt127/VideoCaptionerCripASR.git
   cd VideoCaptionerCripASR
   ```
2. Set up a Python environment and install VideoCaptioner dependencies
   (refer to the upstream VideoCaptioner project for the required packages).
3. Build / install CrispASR from `_build/CrispASR/` (see `_build/CrispASR/README.md`
   and its build scripts: `build-windows.bat`, `build-cuda` / CMake presets, etc.).
4. Download the required ASR models into the appropriate model directories.

## Components & licenses

- **VideoCaptioner** — see upstream VideoCaptioner project for its license.
- **CrispASR** — see `_build/CrispASR/LICENSE`.

> Note: This is an integration/distribution repository assembled from a local
> install. Some setup steps (runtime, models) must be performed locally because
> those large artifacts are not committed here.
