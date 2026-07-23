# CrispASR Full Backend Sweep (Kaggle)

Runs every ASR + TTS (+ MT) backend sequentially on a Kaggle GPU worker, one
model at a time (downloaded → tested → cleaned up to fit ~20 GB scratch), and
**streams each backend's result to an HF dataset as soon as it finishes** so
interim results survive a crash/timeout and the run is **resumable**.

Canonical script: `tools/kaggle-benchmark-all-backends.py` (this dir holds the
kernel metadata; the push stages a copy of the script + `kaggle_harness.py`).

## Streaming + resume
- Per-backend result → `cstr/crispasr-backend-sweep` dataset at
  `<RUN>/results/<category>__<backend>.json` (category = asr|tts|mt).
- On restart the kernel lists existing files for `<RUN>` and **skips** done
  backends. `CRISPASR_SWEEP_RUN` (default `latest`) is the stable run tag —
  bump it for a clean run. `CRISPASR_SWEEP_REPO` overrides the dataset.
- A combined `<RUN>/summary.json` is written at the end.

## Push (chr1s4)
```bash
export KAGGLE_API_TOKEN=<chr1s4 token>
bash tools/kaggle/full-sweep/push.sh   # stages script + harness, then pushes
```
