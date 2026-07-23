#!/usr/bin/env bash
# Usage: scripts/bump-version.sh <version>   e.g.  scripts/bump-version.sh 0.9.0
#
# Updates VERSION, propagates to Cargo.toml / package.json / pyproject.toml etc.
# via sync-version.py, commits, and creates an annotated tag — all in one step.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>  (e.g. $0 0.9.0)" >&2
    exit 1
fi

VERSION="$1"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "$VERSION" > VERSION
python scripts/sync-version.py

# Stage only the files sync-version.py touches (plus VERSION itself).
git add VERSION
for f in \
    bindings/javascript/package.json \
    python/pyproject.toml \
    crispasr/Cargo.toml \
    crispasr-sys/Cargo.toml \
    flutter/crispasr/pubspec.yaml; do
    [ -f "$f" ] && git add "$f" || true
done

git commit -m "release: bump VERSION to $VERSION"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo ""
echo "Created commit + annotated tag v$VERSION."
echo "Push with:  git push && git push origin v$VERSION"
