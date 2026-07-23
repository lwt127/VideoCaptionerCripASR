# CrispASR — MOSS-TTS-Local 4B STOP-HEAD diagnostic (#249, P5 debug) — CPU-only
#
# run1/run2 proved the codec decode is correct (ASR overlap 1.0) but the C++
# generation RUNS AWAY (binary stop head never fires) for short text + Q4_K; the
# stop logit sits at continue~8 / stop~-3 every frame. Static inspection shows
# the C++ port matches the HF reference structurally. THE decisive question:
#   Does the HF reference MossTTSLocal.generate() STOP for "Hello world"?
#
# This runs the HF reference only (no C++, no codec, no ASR) so it needs NO GPU
# (GPU weekly quota is exhausted) — a CPU kernel. It hooks local_text_lm_head and
# runs generate(do_sample=True, text_temperature=1.0) on SHORT + LONG text,
# reporting: frames, stopped?, where the stop logit first exceeds continue, and
# the [continue, stop] trajectory.
#
# Verdict:
#   ref stops short  -> C++ has a subtle bug (compare trajectories next).
#   ref runs away too -> model/param behavior; fix = generation handling
#                        (e.g. text-conditioned max frames, or the "- Tokens:"
#                        prompt field, or sampling config).

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Kaggle sets =1 but the pkg isn't installed
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/moss-tts-local-4b")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
MAXF = int(os.environ.get("MOSS_MAXF", "256"))
SEED = int(os.environ.get("MOSS_SEED", "1234"))

SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage.")
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def run_reference(hf_token):
    # huggingface_hub freezes HF_HUB_ENABLE_HF_TRANSFER at import (Kaggle sets =1 but
    # the pkg is absent) -> force it off on the already-imported module.
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
    try:
        import huggingface_hub.file_download as _fd
        _fd.HF_HUB_ENABLE_HF_TRANSFER = False
    except Exception:  # noqa: BLE001
        pass
    import torch
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(os.cpu_count() or 4)
    log("load reference model (float32, cpu)")
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                      token=hf_token).eval()
    proc = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)  # public; token kwarg rejected by custom proc
    log("model + processor loaded")

    out = {}
    for tag, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
        stop_logits = []

        def hook(_m, _i, o):
            t = o.detach().float().reshape(-1, o.shape[-1])
            stop_logits.append(t[-1].tolist())  # [continue, stop] (candidate order)

        h = model.local_text_lm_head.register_forward_hook(hook)
        try:
            msg = proc.build_user_message(text=text)
            feat = proc([msg], mode="generation")
            input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
            # log the rendered prompt text (compare to the C++ mtl_build_prompt)
            try:
                prompt_txt = proc.tokenizer.decode(input_ids.reshape(-1, input_ids.shape[-1])[:, 0].tolist())
                (RESULTS / f"ref_prompt_{tag}.txt").write_text(prompt_txt)
            except Exception:  # noqa: BLE001
                pass
            torch.manual_seed(SEED)
            t0 = time.time()
            model.generate(input_ids=input_ids, max_new_frames=MAXF, do_sample=True,
                           text_temperature=1.0, temperature=1.0, top_p=0.95, top_k=50)
            n = len(stop_logits)
            stopped = n < MAXF
            cross = next((i for i, (cl, sl) in enumerate(stop_logits) if sl >= cl), None)
            traj = [(i, round(cl, 3), round(sl, 3)) for i, (cl, sl) in enumerate(stop_logits)]
            out[tag] = {"frames": n, "stopped": stopped, "stop_crosses_at": cross,
                        "min_gap": round(min((cl - sl for cl, sl in stop_logits), default=0), 3),
                        "elapsed_s": round(time.time() - t0, 1),
                        "logit_first": traj[:8], "logit_last": traj[-6:]}
            log(f"REF {tag}: frames={n} stopped={stopped} cross_at={cross} min_gap={out[tag]['min_gap']}")
        except Exception as e:  # noqa: BLE001
            out[tag] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1800:]}
            log(f"REF {tag} ERROR: {e}")
        finally:
            h.remove()
    return out


def main():
    summary = {"ref_branch": CRISPASR_REF, "seed": SEED, "maxf": MAXF, "hf_model": HF_MODEL}
    log(f"clone {CRISPASR_REF} (for kaggle_harness)")
    if not REPO.exists():
        import subprocess
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                               CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    hf_token = kh.resolve_hf_token()

    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers==4.57.6", "accelerate", "huggingface_hub", "hf_transfer"])
    try:
        summary["reference"] = run_reference(hf_token)
    except Exception as e:  # noqa: BLE001
        summary["reference"] = {"fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
        log(f"reference fatal: {e}")

    ref = summary.get("reference", {})
    rs = ref.get("short") if isinstance(ref, dict) else None
    if isinstance(rs, dict) and "stopped" in rs:
        summary["verdict"] = ("MODEL_BEHAVIOR: ref ALSO runs away on 'Hello world' -> "
                              "fix = generation handling, not the port"
                              if not rs["stopped"] else
                              "C++_BUG: ref STOPS 'Hello world' at frame %s but C++ runs away -> "
                              "compare trajectories / find the port divergence" % rs.get("frames"))
    else:
        summary["verdict"] = "INCONCLUSIVE (reference error — see tb)"
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"VERDICT: {summary['verdict']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
