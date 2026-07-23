# CrispASR v0.8.18

121 commits since v0.8.17. Headline: **three new backends and two new task
surfaces** — RVC voice conversion, Beat This! beat tracking, and TabCNN guitar
tablature — plus a piano-transcription verb, a Beatrice pitch-estimator port,
and a round of build/release and CI-integrity fixes. Four community PRs merged.

Kept as a patch release: everything below is **additive**. No API breaks, no
changed defaults, no removed flags — existing callers upgrade untouched.

## New backend — RVC voice conversion (§CB1)

Retrieval-based Voice Conversion ported end to end: `enc_p` + `flow` + `dec`,
all in ggml.

- **48 stages at cos 1.00000000** against the torch reference, built up in
  stages (numpy spec → `enc_p` 26 stages → flow 30 → full 47 → `convert()` 48).
- **Noise replay** — the two live RNG sites are injectable, so the whole
  generator is deterministic under test while production `convert()` stays
  random by design. This is what made per-stage parity possible at all.
- Session C ABI wired, so `convert()` is reachable from every binding, not just
  the CLI.
- ⚠️ **f16 is recorded as unsupported**: the relative-position tables must be
  cast to F32. Documented rather than silently degraded.

## New backend — Beat This! beat/downbeat tracking (`--beats`)

CPJKU's ISMIR 2024 beat tracker (§251b), complete through the 12-point checklist.

- `--beats` prints `time_sec<TAB>beat|downbeat` per line, `--beats-format json`
  for JSON.
- **MIT for code AND weights**, and critically **no DBN**: postprocessing is
  peak-picking only, so unlike most beat trackers it carries none of madmom's
  patented, non-commercially-licensed Dynamic Bayesian Network.
- Full surface: session C ABI, Dart binding, registry entry, published weights.
- Fixed a **heap overflow** in the front end found during bring-up.

## New backend — TabCNN guitar tablature (`--tab`)

An **emission scorer**, deliberately not a tablature generator.

- `--tab` prints a per-frame, per-string fret grid; `--tab-format json` adds
  per-string confidences.
- The model emits six independent softmaxes over 21 fret classes per frame with
  **no decoder** — no inter-string coupling, no temporal model, no search. The
  constrained Viterbi/DP that turns scores into a playable fingering (one note
  per string, fret range, capo, hand span) belongs to the caller, via
  `crispasr_session_tab_emissions()`. The CLI's displayed frets are a plain
  argmax and ignore every playability constraint; that is stated at every
  surface because it is the one thing a consumer can get wrong silently.
- **Validated end to end**, not just per-stage: tablature F1 **0.7732** against
  EGSet12 JAMS ground truth vs the torch reference's 0.7708 (Δ +0.0024, argmax
  agreement 98.57 %). `crispasr-diff tabcnn` runs the full pipeline **from the
  waveform** rather than replaying features, because the CQT lives outside the
  network and a feature-replaying diff would never test it.
- Weights **CC BY 4.0** (`cstr/tabcnn-GGUF`, from the EGSet12 record) —
  commercial use permitted, attribution required.
- **Quantization finding:** only two tensors are quantizable at all, and
  preserving `head.weight` is the whole story — quantizing the output layer
  costs 5.8 F1 points at Q4_0 while `dense0` costs nothing. `crispasr-quantize`
  now encodes this. K-quants are impossible here (nothing is 256-aligned), so
  `--q4_k` falls back to Q4_0.

## New task surface — `--piano`

Piano transcription gains its own CLI verb, and a real bug came with it:
**`BatchNorm2d` eps is 0.01, not 1e-5** — upstream passes `momentum`
positionally, where the second positional parameter is `eps`, so all 33
`BatchNorm2d` layers run at 0.01 while the four `BatchNorm1d` keep 1e-5. With
running variances ~0.003 that is a 2.09× error per layer. Reproduced
deliberately, since the weights were trained with the slip.

## Beatrice pitch estimator (§CB2)

Converter, validated reference dump, and a ggml port at **30 stages + e2e, 0
failed**.

## Community PRs merged

- **#286 / #287 (sidon)** — the relative-position-bias rework is a real ~33 %
  predictor-workspace win and the bounded DAC decode is 5.7×; both were kept.
  Follow-up fixes: the leading inference pad was never cropped (results came out
  delayed and lost the same amount of real audio off the end), and DAC graph
  reuse across chunks was silently corrupting joins. Chunked output is now
  asserted **bit-exact** against a whole-utterance decode rather than by a global
  cosine, which a local join failure barely moves.
- **#288 (registry)** — canonical default bundles across the bindings.
- **#289 (voxcpm2 VAE)** — the S2S upscaler, plus its CUDA guard, which had never
  run on CUDA. Now validated: 18/21 checks pass on a P100, the >500000-sample
  path is genuinely exercised, and CPU↔CUDA parity is exact.

## Build, release and CI integrity

- **Fixed the broken release tarball** — fetched ogg/opus are now forced STATIC,
  and the `@rpath` closure is gated. This was the real cause.
- **`bindings/go` cgo LDFLAGS** had been out of sync since the beat-this landing;
  the CI drift check was red on `main` and is now green.
- **`tools/check-registry-urls.py`** (new) — the wiring audit checks a registry
  row *exists*, never that its URL resolves, so an entry pointing at a
  non-existent HuggingFace repo could ship. Now all 226 download URLs are swept;
  gated repos are reported separately rather than failing.
- **`tools/check-kaggle-harness-sync.py`** (new) — every `tools/kaggle/<kernel>/`
  bundles its own `kaggle_harness.py` as the clone-failure fallback, and nothing
  kept them in sync: there were **four distinct versions across 53 files**.
- **Backend wiring audit is now in CI** — it was documented but never actually
  run by any workflow.
- **Kaggle ccache was self-perpetuatingly broken.** `CCACHE_DIR` lived inside
  `/kaggle/working`, which is the output mount that pages at 500 files, so the
  loose cache tree buried `ccache.tar`; refreshing the dataset downloaded a
  truncation and re-uploaded it as the seed. Both account copies were broken this
  way, and the `ccache.tar` warm path had *never* worked — it extracted one level
  too deep, so ccache saw an empty cache. Fixed, and both datasets rebuilt with
  the full 2974-object cache.
- **mel-band-roformer** was CLI-only and unreachable from the session ABI; the
  audit now catches backends missing from the **shipped library**, not just the
  source.
- **Metal teardown** warns instead of `SIGABRT` when resources outlive the
  device.
- **Fetched opus/ogg are now built PIC.** Forcing them STATIC (above) also made
  them non-PIC, and `crispasr-lib` is frequently a shared object, so `ld`
  rejected the link outright (`relocation R_X86_64_PC32 against
  'silk_LBRR_flags_iCDF_ptr' ... recompile with -fPIC`). Invisible on macOS,
  where everything is position-independent anyway — so the STATIC fix looked
  clean locally and only broke on Linux.
- **`miotts`** used `substr` to assign a prefix of a string to itself; now
  `resize()`.
- **The wiring audit named the wrong failure.** It printed
  `RESULT: FAIL (required gap)` for any of four independent causes, so a run
  whose only problem was Go LDFLAGS drift reported a required *wiring* gap two
  lines below `✅ REQUIRED wiring: ...`. It now names the actual cause.

A process note worth recording, since it shaped this release: `main` took ~120
commits in one day, and CI runs were repeatedly cancelled by the next push
(`concurrency: cancel-in-progress`). **Every Lint run on `main` that day was
cancelled**, which hid the `miotts` finding for two days — a cancelled check
looks much like a passing one in the runs list. This release was cut from a tree
verified green job-by-job, not from a run summary.

## Registry

crepe quants are now listed (they were on HF but unreachable by `-m auto`), BTC
gains q8_0, and BTC q4_k is **refused on evidence** — 95.46 % tetrads for 0.6 MB
saved.

## Upgrading

No API breaks. New entry points are additive: `crispasr_session_tab*` (6),
`crispasr_session_beats*`, and the RVC `convert()` dispatch. `--tab`, `--beats`
and `--piano` are new verbs; existing flags are unchanged. Nothing in this
release requires a caller to change anything.

One behaviour change worth knowing about if you use **sidon**: `--s2s` output is
now correctly time-aligned (the leading inference pad was previously never
cropped, so results were delayed and lost the same amount of audio off the end),
and the DAC decode is chunked by default, bounded at ~0.79 GiB instead of
~4.5 GiB at 55 s. Output is bit-exact against the whole-utterance path; set
`CRISPASR_SIDON_DECODER_CHUNK_FRAMES=0` to restore the single-graph decode.
