"""CrispASR 后端 / 模型 / VAD 目录（单一数据源）。

数据依据 CrispASR 源码 `src/crispasr_model_registry.cpp`（k_registry[]）、
`examples/cli/crispasr_backend.cpp`、`examples/cli/crispasr_vad_cli.cpp`。

- 每个后端使用 `--backend <value>` 选择；
- 模型使用 `-m <auto|filename>` 选择（auto 表示首次运行自动下载该后端默认模型）；
- VAD 方法使用 `--vad -vm <keyword>` 选择。
"""

CRISP_ASR_PROJECT_URL = "https://github.com/CrispStrobe/CrispASR"

# VAD 方法关键字（传给 -vm/--vad-model）
CRISP_ASR_VAD_METHODS = [
    {"label": "Silero (默认, 标准, ~885KB)", "value": "silero"},
    {"label": "FireRedVAD (推荐, F1=97.57%, 2.4MB)", "value": "firered"},
    {"label": "MarbleNet (最小, 439KB, 6 语言)", "value": "marblenet"},
    {"label": "Whisper-VAD (实验性, 22MB, 较慢)", "value": "whisper-vad"},
]

# 后端引擎目录：每个后端含友好标签、--backend 值，以及可选模型列表。
# 模型项：label 为下拉显示文本，value 为 -m 传入值（auto 或具体文件名）。
CRISP_ASR_BACKENDS = [
    {
        "label": "Parakeet (NeMo, 快, 多语言, 词级时间戳)",
        "backend": "parakeet",
        "models": [
            {
                "label": "TDT 0.6B v3 (多语言, 默认, 自动下载)",
                "value": "auto",
                "backend": "parakeet",
            },
            {
                "label": "TDT 0.6B v2 (英语)",
                "value": "auto",
                "backend": "parakeet-v2",
            },
            {
                "label": "TDT 1.1B (英语, 更大)",
                "value": "auto",
                "backend": "parakeet-tdt-1.1b",
            },
            {
                "label": "TDT-CTC 110M (英语, 最小)",
                "value": "auto",
                "backend": "parakeet-tdt_ctc-110m",
            },
            {
                "label": "TDT-CTC 1.1B (英语, 含标点)",
                "value": "auto",
                "backend": "parakeet-tdt_ctc-1.1b",
            },
            {
                "label": "TDT 0.6B (日语)",
                "value": "auto",
                "backend": "parakeet-ja",
            },
        ],
    },
    {
        "label": "Fun-ASR Nano (中粤英日韩)",
        "backend": "funasr",
        "models": [
            {"label": "Fun-ASR-Nano (自动下载)", "value": "auto", "backend": "funasr"},
            {
                "label": "Fun-ASR-MLT-Nano (31 语言)",
                "value": "auto",
                "backend": "fun-asr-mlt-nano",
            },
        ],
    },
    {
        "label": "SenseVoice (50+ 语言, 含情感)",
        "backend": "sensevoice",
        "models": [
            {
                "label": "SenseVoice-Small (自动下载)",
                "value": "auto",
                "backend": "sensevoice",
            },
            {
                "label": "SenseVoice-Small Q8",
                "value": "sensevoice-small-q8_0.gguf",
                "backend": "sensevoice",
            },
        ],
    },
    {
        "label": "Paraformer-zh (中英, 非自回归)",
        "backend": "paraformer",
        "models": [
            {
                "label": "Paraformer-zh (自动下载)",
                "value": "auto",
                "backend": "paraformer",
            },
            {
                "label": "Paraformer-zh Q8",
                "value": "paraformer-zh-q8_0.gguf",
                "backend": "paraformer",
            },
        ],
    },
    {
        "label": "FireRedASR2 (普通话 + 方言)",
        "backend": "firered-asr",
        "models": [
            {
                "label": "FireRedASR2-AED (自动下载)",
                "value": "auto",
                "backend": "firered-asr",
            },
        ],
    },
    {
        "label": "GLM-ASR Nano (17 语言, 含粤语)",
        "backend": "glm-asr",
        "models": [
            {"label": "GLM-ASR-Nano (自动下载)", "value": "auto", "backend": "glm-asr"},
        ],
    },
    {
        "label": "Mega-ASR (Qwen3-1.7B, 抗噪)",
        "backend": "mega-asr",
        "models": [
            {"label": "Mega-ASR-1.7B (自动下载)", "value": "auto", "backend": "mega-asr"},
        ],
    },
    {
        "label": "Voxtral Mini 3B (语音大模型, 8 语言)",
        "backend": "voxtral",
        "models": [
            {"label": "Voxtral-Mini-3B (自动下载)", "value": "auto", "backend": "voxtral"},
        ],
    },
    {
        "label": "Voxtral 4B Realtime (13 语言, 流式)",
        "backend": "voxtral4b",
        "models": [
            {
                "label": "Voxtral-Mini-4B-Realtime (自动下载)",
                "value": "auto",
                "backend": "voxtral4b",
            },
        ],
    },
    {
        "label": "Gemma4-E2B (140+ 语言, 需 HF 令牌)",
        "backend": "gemma4-e2b",
        "models": [
            {
                "label": "Gemma4-E2B-IT (自动下载)",
                "value": "auto",
                "backend": "gemma4-e2b",
            },
        ],
    },
    {
        "label": "omniASR-LLM (1600+ 语言)",
        "backend": "omniasr-llm",
        "models": [
            {
                "label": "omniASR-LLM 300M v2 (自动下载)",
                "value": "auto",
                "backend": "omniasr-llm",
            },
            {
                "label": "omniASR-LLM 1B",
                "value": "auto",
                "backend": "omniasr-llm-1b",
            },
        ],
    },
    {
        "label": "Whisper (通用, ggml 模型)",
        "backend": "whisper",
        "models": [
            {"label": "ggml-tiny", "value": "ggml-tiny.bin", "backend": "whisper"},
            {"label": "ggml-base", "value": "ggml-base.bin", "backend": "whisper"},
            {"label": "ggml-small", "value": "ggml-small.bin", "backend": "whisper"},
            {"label": "ggml-medium", "value": "ggml-medium.bin", "backend": "whisper"},
            {
                "label": "ggml-large-v3",
                "value": "ggml-large-v3.bin",
                "backend": "whisper",
            },
        ],
    },
]


def get_backend_labels() -> list[str]:
    return [b["label"] for b in CRISP_ASR_BACKENDS]


def get_backend_by_label(label: str) -> dict | None:
    for b in CRISP_ASR_BACKENDS:
        if b["label"] == label:
            return b
    return None


def get_model_labels(backend_label: str) -> list[str]:
    b = get_backend_by_label(backend_label)
    if not b:
        return []
    return [m["label"] for m in b["models"]]


def resolve_model(backend_label: str, model_label: str) -> tuple[str, str]:
    """根据后端标签与模型标签，返回 (实际 --backend 值, -m 值)。

    模型项可携带自己的 backend（如 Parakeet 的多个变体对应不同 --backend）。
    """
    b = get_backend_by_label(backend_label)
    if not b:
        return ("whisper", "auto")
    for m in b["models"]:
        if m["label"] == model_label:
            return (m.get("backend", b["backend"]), m["value"])
    # 回退到该后端第一个模型
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
