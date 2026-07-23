# CrispASR v0.8.19

Patch release on top of v0.8.18. The headline is a **canary-qwen long-audio
fix (#290)** that also removes a multi-gigabyte memory blowup.

For everything in the v0.8.18 cycle — RVC voice conversion, Beat This! beat
tracking, TabCNN guitar tablature, the `--piano` verb, four merged community
PRs — see `RELEASE_NOTES_v0.8.18.md`. Nothing there changed.

## Fixed — canary-qwen ignored chunking and grew to 10 GiB (#290)

`canary-qwen` stopped chunking long audio **with or without `--chunk-seconds`**,
produced near-empty transcripts, and its memory scaled with clip length. The
reporter measured, on `canary-qwen-2.5b-q8_0` with the Vulkan backend:

| clip | RSS |
|:--|--:|
| tot4.wav | 384 MiB |
| tot5.wav | 6300 MiB |
| tot7.wav | 10203 MiB |

**Root cause: the backend declared a capability it does not implement.**
`CAP_INTERNAL_CHUNKING` asserts that a backend slices long audio itself.
`src/canary_qwen.cpp` has no chunking code at all — unlike `src/parakeet.cpp`
and `src/canary.cpp`, which take a real `chunk_seconds`. Those two earn the
flag; canary-qwen never did.

That single flag disabled **both** dispatcher safety nets, which is why it
reproduced either way:

- **with `--chunk-seconds`** — the #257 gate handed the whole clip to the
  backend, trusting an internal chunker that does not exist;
- **without it** — the 30 s long-audio fallback bailed out at "backend handles
  its own chunking", so it never armed.

canary-qwen also does not opt into the auto-VAD safeguard, so every net keyed
off the same flag. The result was one full-length FastConformer pass: O(T²)
attention (matching the memory curve above) with output degraded well past the
encoder's trained window.

Commit `1a2b3dcea` (#257) was written for parakeet and is correct there — it
keys on the capability, so the defect was always the declaration, not the gate.
Overlap-save context remains blocked for canary-qwen separately (#218); that
gate is correct and unrelated.

**Verification.** Confirmed via the capability bitmask, with parakeet, canary
and fastconformer-ctc as controls: they still assert the flag, canary-qwen no
longer does. Full unit suite passes (1076/1076).

**Not yet verified against the reporter's audio.** The model is 4.08 GB and the
bug's signature is 6–10 GiB of RSS, which did not fit on the machine available
when the fix was written. The reasoning and the bitmask check are solid, but if
you hit #290, please confirm on your own long files and reopen if anything
remains.

## Also in this release

- **`crispasr-f0-eval`** — new tool to compare F0 backends on the shipped C++
  path rather than on a Python reference.
- **Beatrice PitchEstimator** accuracy measured; a wrong claim in
  `src/beatrice_pitch.h` corrected.

## Upgrading

Drop-in. No API changes, no changed defaults, no removed flags. If you use
`canary-qwen` on audio longer than ~30 s, upgrading is strongly recommended.
