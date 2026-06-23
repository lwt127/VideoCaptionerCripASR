"""End-to-end CrispASR pipeline test through VideoCaptioner's transcribe()."""
import sys
import time

from videocaptioner.core.asr.transcribe import transcribe
from videocaptioner.core.entities import TranscribeConfig, TranscribeModelEnum


def progress(p, msg):
    print(f"  [{p:3d}%] {msg}", flush=True)


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else "jfk.wav"
    backend = sys.argv[2] if len(sys.argv) > 2 else "moonshine"
    model = sys.argv[3] if len(sys.argv) > 3 else "auto"

    cfg = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.CRISP_ASR,
        transcribe_language="en",
        need_word_time_stamp=False,
        crisp_asr_backend=backend,
        crisp_asr_model=model,
        crisp_asr_use_vad=False,  # jfk is short; skip VAD for a faster test
    )
    print(cfg.print_config())
    print(f"\n>>> Transcribing '{audio}' with backend={backend} model={model}\n")

    t0 = time.time()
    result = transcribe(audio, cfg, callback=progress)
    dt = time.time() - t0

    print(f"\n=== RESULT ({len(result.segments)} segments, {dt:.1f}s) ===")
    for seg in result.segments:
        print(f"[{seg.start_time/1000:6.2f} -> {seg.end_time/1000:6.2f}] {seg.text}")

    print("\n=== SRT ===")
    print(result.to_srt())

    if not result.segments:
        print("\n!!! NO SEGMENTS — pipeline produced empty output", flush=True)
        sys.exit(1)
    print("\nOK: end-to-end pipeline succeeded.")


if __name__ == "__main__":
    main()
