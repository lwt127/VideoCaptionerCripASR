"""CrispASR 后端 / 模型 / VAD 目录（单一数据源）。

数据依据 CrispASR 源码 `src/crispasr_model_registry.cpp`（k_registry[]）、
`examples/cli/crispasr_backend.cpp`、`examples/cli/crispasr_vad_cli.cpp`。

- 每个后端使用 `--backend <value>` 选择；
- 模型使用 `-m <auto|filename>` 选择（auto 表示首次运行时自动下载该后端默认模型，
  下载到 ~/.cache/crispasr）；
- VAD 方法使用 `--vad -vm <keyword>` 选择。

包含全部 ASR 后端（不含纯 TTS / 对齐器 / LID / VAD 辅助行），
另含 m2m100 / madlad 等文本翻译后端。
"""

CRISP_ASR_PROJECT_URL = "https://github.com/CrispStrobe/CrispASR"
CRISP_ASR_REPO = "CrispStrobe/CrispASR"


# VAD 方法关键字（传给 -vm/--vad-model）
CRISP_ASR_VAD_METHODS = [
    {"label": "Silero (默认, 标准, ~885KB)", "value": "silero"},
    {"label": "FireRedVAD (推荐, F1=97.57%, 2.4MB)", "value": "firered"},
    {"label": "MarbleNet (最小, 439KB, 6 语言)", "value": "marblenet"},
    {"label": "Whisper-VAD (实验性, 22MB, 较慢)", "value": "whisper-vad"},
]


def _auto(label: str, backend: str, value: str = "auto") -> dict:
    return {"label": label, "value": value, "backend": backend}


# 后端引擎目录（全部 ASR 后端）。
# 同一“识别架构”家族（如 Parakeet）下，多个变体合并到同一后端项的模型下拉中；
# 其余每个后端单独成项，模型项默认“自动下载”。
CRISP_ASR_BACKENDS = [
    {
        # 推荐：非自回归, 快且稳定, 适合长音频 (CPU 也流畅), 50+ 语言含日语
        "label": "SenseVoice (✨推荐✨, 50+ 语言, 快, 稳定)",
        "backend": "sensevoice",
        "models": [
            _auto("SenseVoice Small (自动下载)", "sensevoice"),
            {
                "label": "SenseVoice Small Q8",
                "value": "sensevoice-small-q8_0.gguf",
                "backend": "sensevoice",
            },
        ],
    },
    {
        "label": "Paraformer-zh (中英, 非自回归, 稳定)",
        "backend": "paraformer",
        "models": [
            _auto("Paraformer-zh (自动下载)", "paraformer"),
            {
                "label": "Paraformer-zh Q8",
                "value": "paraformer-zh-q8_0.gguf",
                "backend": "paraformer",
            },
        ],
    },
    {
        "label": "Whisper (通用, 99 语言, ggml)",
        "backend": "whisper",
        "models": [
            {"label": "ggml-tiny", "value": "ggml-tiny.bin", "backend": "whisper"},
            {"label": "ggml-base (默认)", "value": "ggml-base.bin", "backend": "whisper"},
            {"label": "ggml-small", "value": "ggml-small.bin", "backend": "whisper"},
            {"label": "ggml-medium", "value": "ggml-medium.bin", "backend": "whisper"},
            {
                "label": "ggml-large-v3",
                "value": "ggml-large-v3.bin",
                "backend": "whisper",
            },
            {
                # kotoba-whisper v2.2（日语微调）的 ggml 量化版，可在 whisper 后端运行。
                # 该模型不在 CrispASR 内置注册表中，需通过 --hf-repo 从自定义仓库下载，
                # 由 crisp_asr.py 的 CRISP_ASR_WHISPER_HF_REPOS 映射处理。
                "label": "kotoba-whisper-v2.2 ✨ (日语, ggml q8_0)",
                "value": "kotoba-whisper-v2.2-ggml-q8_0.bin",
                "backend": "whisper",
            },
        ],
    },
    {
        # 注意：Parakeet 的所有变体都使用同一个 --backend "parakeet"，
        # 具体变体通过 -m <registry-key> 选择（如 -m parakeet-ja）。
        # CrispASR 不接受 --backend parakeet-ja（会报 "unknown backend"），
        # 因此这些变体的 backend 必须保持 "parakeet"，变体名放在 -m 值里。
        "label": "Parakeet (NeMo, 快, 多语言, 词级时间戳)",
        "backend": "parakeet",
        "models": [
            _auto("TDT 0.6B v3 (多语言, 默认, 自动下载)", "parakeet", "auto"),
            _auto("TDT 0.6B v2 (英语)", "parakeet", "parakeet-v2"),
            _auto("TDT 0.6B (日语)", "parakeet", "parakeet-ja"),
            _auto("TDT 1.1B (英语, 更大)", "parakeet", "parakeet-tdt-1.1b"),
            _auto("TDT-CTC 110M (英语, 最小)", "parakeet", "parakeet-tdt_ctc-110m"),
            _auto("TDT-CTC 1.1B (含标点, 多语言)", "parakeet", "parakeet-tdt_ctc-1.1b"),
            _auto("RNNT 0.6B (英语)", "parakeet", "parakeet-rnnt-0.6b"),
            _auto("RNNT 1.1B (英语)", "parakeet", "parakeet-rnnt-1.1b"),
            _auto("CTC 0.6B (英语)", "parakeet", "parakeet-ctc-0.6b"),
            _auto("CTC 1.1B (英语)", "parakeet", "parakeet-ctc-1.1b"),
        ],
    },
    {
        "label": "FastConformer-CTC (NeMo, 英语)",
        "backend": "fastconformer-ctc",
        "models": [_auto("FastConformer-CTC Large (自动下载)", "fastconformer-ctc")],
    },
    {
        "label": "Canary (NVIDIA, 25 欧语, 含翻译)",
        "backend": "canary",
        "models": [_auto("Canary 1B v2 (自动下载)", "canary")],
    },
    {
        "label": "Voxtral Mini 3B (Mistral, 8 语言)",
        "backend": "voxtral",
        "models": [_auto("Voxtral Mini 3B (自动下载)", "voxtral")],
    },
    {
        "label": "Voxtral 4B Realtime (13 语言, 流式)",
        "backend": "voxtral4b",
        "models": [_auto("Voxtral Mini 4B Realtime (自动下载)", "voxtral4b")],
    },
    {
        "label": "Granite Speech (IBM)",
        "backend": "granite",
        "models": [
            _auto("Granite 4.0 1B (自动下载)", "granite"),
            _auto("Granite 4.1 2B", "granite-4.1"),
            _auto("Granite 4.1 2B Plus (含标点)", "granite-4.1-plus"),
            _auto("Granite 4.1 2B NAR (非自回归)", "granite-4.1-nar"),
        ],
    },
    {
        "label": "Qwen3-ASR (30 语言 + 22 中文方言)",
        "backend": "qwen3",
        "models": [
            _auto("Qwen3-ASR 0.6B (自动下载)", "qwen3"),
            _auto("Qwen3-ASR 1.7B", "qwen3-1.7b"),
        ],
    },
    {
        "label": "Mega-ASR (Qwen3-1.7B, 抗噪)",
        "backend": "mega-asr",
        "models": [_auto("Mega-ASR 1.7B (自动下载)", "mega-asr")],
    },
    {
        "label": "Fun-ASR Nano (中粤英日韩)",
        "backend": "funasr",
        "models": [
            _auto("Fun-ASR Nano (自动下载)", "funasr"),
            _auto("Fun-ASR MLT Nano (31 语言)", "fun-asr-mlt-nano"),
        ],
    },
    {
        "label": "Cohere Transcribe (13 语言)",
        "backend": "cohere",
        "models": [
            _auto("Cohere Transcribe (自动下载)", "cohere"),
            {
                "label": "Cohere ASR 日语",
                "value": "cohere-asr-ja-q4_k.gguf",
                "backend": "cohere",
            },
        ],
    },
    {
        "label": "wav2vec2 / HuBERT / data2vec (自监督 CTC)",
        "backend": "wav2vec2",
        "models": [
            _auto("wav2vec2 XLSR (英语, 自动下载)", "wav2vec2"),
            _auto("wav2vec2 XLSR-53 (德语)", "wav2vec2-de"),
            _auto("HuBERT Large (英语)", "hubert"),
            _auto("data2vec Audio (英语)", "data2vec"),
        ],
    },
    {
        "label": "omniASR (1600+ 语言)",
        "backend": "omniasr",
        "models": [
            _auto("omniASR CTC 1B v2 (自动下载)", "omniasr"),
            _auto("omniASR CTC 300M v2", "omniasr-300m"),
            _auto("omniASR-LLM 300M v2", "omniasr-llm"),
            _auto("omniASR-LLM 1B", "omniasr-llm-1b"),
        ],
    },
    {
        "label": "FireRedASR2 (普通话 + 方言)",
        "backend": "firered-asr",
        "models": [_auto("FireRedASR2 AED (自动下载)", "firered-asr")],
    },
    {
        "label": "GLM-ASR Nano (17 语言, 含粤语)",
        "backend": "glm-asr",
        "models": [_auto("GLM-ASR Nano (自动下载)", "glm-asr")],
    },
    {
        "label": "Kyutai STT (英/法)",
        "backend": "kyutai-stt",
        "models": [_auto("Kyutai STT 1B (自动下载)", "kyutai-stt")],
    },
    {
        "label": "Gemma4-E2B (140+ 语言, 需 HF 令牌)",
        "backend": "gemma4-e2b",
        "models": [_auto("Gemma4-E2B-IT (自动下载)", "gemma4-e2b")],
    },
    {
        "label": "MiMo-ASR (小米, 普通话 + 方言)",
        "backend": "mimo-asr",
        "models": [_auto("MiMo-ASR (自动下载, ~4.2GB)", "mimo-asr")],
    },
    {
        "label": "MOSS-Audio 4B (中英, 音频理解)",
        "backend": "moss-audio",
        "models": [_auto("MOSS-Audio 4B Instruct (自动下载)", "moss-audio")],
    },
    {
        "label": "VibeVoice ASR (50+ 语言)",
        "backend": "vibevoice",
        "models": [_auto("VibeVoice ASR (自动下载, ~4.5GB)", "vibevoice")],
    },
    {
        "label": "KugelAudio (多语言, ~14GB)",
        "backend": "kugelaudio",
        "models": [_auto("KugelAudio 0 Open F16 (自动下载)", "kugelaudio")],
    },
    {
        "label": "Moonshine (轻量, 英语 + 6 语言)",
        "backend": "moonshine",
        "models": [
            _auto("Moonshine Tiny (自动下载)", "moonshine"),
            _auto("Moonshine Base 德语", "moonshine-de"),
            _auto("Moonshine Tiny 德语", "moonshine-tiny-de"),
            _auto("Moonshine Streaming Tiny (流式)", "moonshine-streaming"),
        ],
    },
]


# 文本翻译后端（text-to-text），非语音识别；如需可在 UI 中单列。
CRISP_ASR_TRANSLATION_BACKENDS = [
    {"label": "m2m100 418M (100 语言)", "backend": "m2m100", "value": "auto"},
    {"label": "m2m100 WMT21 (英↔7 语言)", "backend": "m2m100-wmt21", "value": "auto"},
    {"label": "MADLAD-400 3B (419 语言)", "backend": "madlad", "value": "auto"},
]


def get_backend_labels() -> list[str]:
    return [b["label"] for b in CRISP_ASR_BACKENDS]


def get_backend_by_label(label: str) -> "dict | None":
    for b in CRISP_ASR_BACKENDS:
        if b["label"] == label:
            return b
    return None


def get_model_labels(backend_label: str) -> list[str]:
    b = get_backend_by_label(backend_label)
    if not b:
        return []
    return [m["label"] for m in b["models"]]


def resolve_model(backend_label: str, model_label: str) -> "tuple[str, str]":
    """根据后端标签与模型标签，返回 (实际 --backend 值, -m 值)。

    模型项可携带自己的 backend（如 Parakeet 的多个变体对应不同 --backend）。
    """
    b = get_backend_by_label(backend_label)
    if not b:
        return ("whisper", "auto")
    for m in b["models"]:
        if m["label"] == model_label:
            return (m.get("backend", b["backend"]), m["value"])
    if b["models"]:
        first = b["models"][0]
        return (first.get("backend", b["backend"]), first["value"])
    return (b["backend"], "auto")


def get_vad_labels() -> list[str]:
    return [v["label"] for v in CRISP_ASR_VAD_METHODS]


def resolve_vad_method(vad_label: str) -> str:
    for v in CRISP_ASR_VAD_METHODS:
        if v["label"] == vad_label:
            return v["value"]
    return "silero"
