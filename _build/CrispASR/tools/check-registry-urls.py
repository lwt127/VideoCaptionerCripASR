#!/usr/bin/env python3
"""Verify every model-registry URL actually resolves.

The backend-wiring audit checks that a registry ROW EXISTS. It cannot check that
the row points at anything real, and nothing else did either — so a registry
entry naming a HuggingFace repo that had never been created shipped to main and
`-m auto` would have 404'd for that backend (tabcnn, 2026-07-20). The failure is
invisible until a user tries to download.

This closes that gap: parse `src/crispasr_model_registry.cpp`, HEAD every URL,
report anything that does not resolve.

    python tools/check-registry-urls.py                 # all entries
    python tools/check-registry-urls.py --backend tabcnn
    python tools/check-registry-urls.py --timeout 20 --jobs 8

Exit 1 if any URL fails. Network-dependent, so this is NOT a unit test — run it
before a release, or on a schedule like the readme-langs lint job.

Deliberately uses HEAD with redirects followed. HuggingFace `resolve/main/...`
URLs 302 to a CDN, so a naive no-redirect check reports 302 for healthy files
and 404 only for genuinely missing ones.
"""

import argparse
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "src" / "crispasr_model_registry.cpp"

# {"name", "file.gguf", "https://…", "~size", companion_file, companion_url, …}
# Entries span lines and contain nullptrs, so match the URL string literals and
# associate each with the nearest preceding backend name.
ENTRY_RE = re.compile(r'\{\s*"([a-zA-Z0-9._+-]+)"\s*,')
URL_RE = re.compile(r'"(https?://[^"\s]+)"')

# Only DOWNLOAD urls are checkable. The registry also embeds reference links
# inside licence TEXT, e.g.
#     "LFM Open License v1.0 (commercial use OK ...; see "
#     "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)"
# Those are prose: they carry a trailing ")" and often point at gated repos that
# answer 401 to an anonymous HEAD. Checking them produced 7 false alarms on the
# first run of this script -- a checker that cries wolf is worse than none, so
# restrict to things that are actually fetched by the downloader.
DOWNLOAD_HINTS = ("/resolve/",)
MODEL_SUFFIXES = (".gguf", ".bin", ".onnx", ".tar", ".zip", ".json", ".txt", ".model")


def is_download_url(url: str) -> bool:
    return any(h in url for h in DOWNLOAD_HINTS) or url.endswith(MODEL_SUFFIXES)


def parse_registry(path: Path):
    """Yield (backend, url) in file order."""
    text = path.read_text(errors="replace")
    out, current = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue  # a commented-out entry is not shipped
        m = ENTRY_RE.search(line)
        if m:
            current = m.group(1)
        for um in URL_RE.finditer(line):
            url = um.group(1).rstrip(").,;")
            if is_download_url(url):
                out.append((current or "?", url))
    return out


def check(url: str, timeout: float):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "crispasr-registry-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        # Some hosts reject HEAD but serve GET. Retry once before failing, or a
        # healthy URL gets reported broken.
        if e.code in (403, 405, 501):
            try:
                g = urllib.request.Request(
                    url, headers={"User-Agent": "crispasr-registry-check", "Range": "bytes=0-0"})
                with urllib.request.urlopen(g, timeout=timeout) as r2:
                    return r2.status, "(via GET)"
            except Exception as e2:
                return e.code, f"HEAD {e.code}, GET {type(e2).__name__}"
        return e.code, str(e.reason)[:60]
    except Exception as e:
        return 0, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", help="only check entries for this backend")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if not REGISTRY.is_file():
        print(f"ERROR: {REGISTRY} not found", file=sys.stderr)
        return 1
    pairs = parse_registry(REGISTRY)
    if args.backend:
        pairs = [p for p in pairs if p[0] == args.backend]
    # De-duplicate: several entries legitimately share a URL (aliases).
    seen, uniq = set(), []
    for b, u in pairs:
        if u not in seen:
            seen.add(u)
            uniq.append((b, u))

    print(f"checking {len(uniq)} unique URLs from {REGISTRY.relative_to(ROOT)}\n")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda p: (p[0], p[1], *check(p[1], args.timeout)), uniq))

    bad, gated = [], []
    for backend, url, status, note in sorted(results, key=lambda r: r[0]):
        # 401/403 on a download URL means the repo is GATED, not missing. That is
        # a real state for several entries (mistralai, LiquidAI, funasr) and the
        # registry handles it by printing the licence gate -- so report it,
        # don't fail on it.
        if status in (401, 403):
            gated.append((backend, url))
            continue
        ok = 200 <= status < 400
        if not ok:
            bad.append((backend, url, status, note))
            print(f"  FAIL {status:>3} {backend:<28} {url}")
            if note:
                print(f"              {note}")
    if gated:
        print(f"  {len(gated)} gated (401/403 — repo exists, needs auth):")
        for backend, url in gated:
            print(f"    {backend}: {url}")
    print()
    if bad:
        print(f"{len(bad)} of {len(uniq)} registry URLs do NOT resolve:", file=sys.stderr)
        for backend, url, status, _ in bad:
            print(f"  {backend}: HTTP {status} {url}", file=sys.stderr)
        print("\nAn unreachable URL means `-m auto` / --auto-download fails for that\n"
              "backend. Either publish the artifact or remove the entry.", file=sys.stderr)
        return 1
    print(f"OK: all {len(uniq)} registry URLs resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
