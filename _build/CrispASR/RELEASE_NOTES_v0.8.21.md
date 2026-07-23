# CrispASR v0.8.21

Patch release. Headline: **generation-length and sampling params that were
silently ignored are now honored** — across ten ASR backends (`--max-new-tokens`,
#292) and the two MOSS TTS backends (#293) — plus chunk-local speaker scope for
diarization. Everything is additive; no API breaks, no changed defaults.

## Fixed — `--max-new-tokens` honored across 10 ASR backends (#292)

Several ASR backends hardcoded their decode cap, so `--max-new-tokens` did
nothing and a long single-pass run (`--chunk-seconds 0`) truncated — reported
against `moss-diarize`, whose 300 s file stopped at 164 s. The same hardcoded cap
was in nine more: `canary`, `canary-qwen`, `glm-asr`, `funasr`, `mimo-asr`,
`moss-transcribe`, `mini-omni2`, `higgs-stt`, `higgs`. All ten now forward the
value (CLI **and** session C-ABI).

Each backend keeps its **own** former default, so nothing regresses, and the
CLI's global 512 default can never *shrink* a backend whose own default is higher
— the value is forwarded only when you explicitly pass `--max-new-tokens`.
`moonshine` was deliberately left alone: its cap is length-derived and
architectural (a short-form model), not a naive constant.

**CUDA-validated** on the reporter's exact backend (moss-diarize): raising
`--max-new-tokens` 64→4096 raised output 216→366 words (CPU) / 232→382 (CUDA) — a
deterministic, causal increase.

## Fixed — MOSS TTS synthesis params wired (#293, @Bonenk)

`moss-tts` and `moss-tts-local` ignored `--max-new-tokens`, `tts-max-speech-tokens`
(duration), `tts-top-p`, `tts-top-k`, and `tts-repetition-penalty`, leaving the
backends at hardcoded defaults. They now flow through. (A follow-up gated
`max_new_tokens` on the explicit flag so the CLI default no longer caps synthesis
at 512 frames — same discipline as #292.)

## New — chunk-local speaker scope (#292 part 2)

Segments now carry a `chunk_id` in the JSON (present only on multi-chunk runs).
Diarization `(Speaker N)` labels are chunk-local — they restart per chunk — so a
consumer must not assume "Speaker 1" is the same person across different
`chunk_id`s. Segments sharing a `chunk_id` share a speaker numbering. This is the
continuity-vs-ID-swap signal for subtitle pipelines.

## CI / test hygiene

- **cppcheck pinned** to `ubuntu-22.04` in the lint workflow — it was on
  `ubuntu-latest`, so an image flip could change findings and turn Lint red with
  no code change (clang-format was already pinned for this reason).
- **mel-band-roformer smoke test** added — it links the backend, an artifact-level
  guard against the "compiled but dropped from the shipped library" failure.
- **C# task-backend live test** (`Session.Tab`) added, model-gated.

## Upgrading

Drop-in from v0.8.20. If you use `--max-new-tokens` with any ASR backend, or the
sampling/duration params with the MOSS TTS backends, or consume diarized JSON,
this is the release you want.
