#!/bin/bash
# Push under chr1str (CrispASR kernel convention).
# ⚠ If a run of this kernel is already live: `yes | kaggle kernels delete
# chr1str/crispasr-madlad-quants` FIRST — a re-push STACKS a second GPU session
# and both keep burning quota (kaggle_usage gotcha #25).
export KAGGLE_API_TOKEN=KGAT_cb3f25c81b9e65d706ebcf655f1daa42
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
