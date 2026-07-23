# CrispASR v0.8.20

Patch on top of v0.8.19. It completes the library-delivery work: the two GPU lib
bundles that v0.8.19 dropped are restored, and **iOS ships for the first time**.

For the substance of this cycle — the canary-qwen long-audio fix (#290) and the
rpath/flatten fixes that made the desktop lib bundles loadable at all — see
`RELEASE_NOTES_v0.8.19.md`. This release adds two things to that and changes
nothing else.

## Restored — CUDA and HIP Linux lib bundles

v0.8.18 shipped `libcrispasr-linux-x86_64-cuda` and `-hip`, but like every lib
bundle that release they had a broken rpath and could not be loaded. v0.8.19
fixed the rpath and added a gate that actually `dlopen`s each bundle before
shipping it — and that gate then **false-failed** the GPU bundles: it aborted on
`libcuda.so.1`, which is the host's NVIDIA driver, provided at runtime and never
part of any bundle. So v0.8.19 shipped correct CPU/Metal/Vulkan libs but dropped
the two GPU ones entirely.

The gate now separates the two dependency classes it was conflating:

- **bundle-internal** (`libggml`, `libcrispasr`) — must resolve via the bundle's
  own rpath; this is the v0.8.18 bug and is checked statically, no driver needed;
- **external driver** (`libcuda`, `libamdhip64`, …) — correctly absent from the
  bundle, tolerated when it is the *only* thing a confirmatory `dlopen` cannot
  find.

Both GPU bundles are back, and for the first time they are verified loadable
(their internal closure is proven; the driver is resolved on the user's machine).

## New — Apple xcframework (iOS)

iOS had never shipped. The xcframework build existed but lived in a workflow
that is deliberately not triggered on release tags, so its "attach to release"
step was dead code. The build now runs from `release.yml`, where tags live, and
attaches `crispasr-<tag>-xcframework.zip`. It is a **static** framework, so
unlike the desktop shared libraries it has no rpath to resolve — it drops
straight into an Xcode or Flutter project.

Android already shipped a flat `lib/` bundle (all backend + ggml `.so` plus the
header) and is unchanged.

## Upgrading

Drop-in from v0.8.19 or v0.8.18. No API changes. If you consume the CUDA or HIP
Linux lib bundle, or you build for iOS, this is the release you want.
