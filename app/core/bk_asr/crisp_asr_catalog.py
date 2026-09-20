"""CrispASR 后端 / 模型 / VAD 目录（单一数据源）。

数据依据 CrispASR 源码 `src/crispasr_model_registry.cpp`（k_registry[]）、
`examples/cli/crispasr_backend.cpp`、`examples/cli/crispasr_vad_cli.cpp`。

- 每个后端使用 `--backend <value>` 选择；
- 模型使用 `-m <auto|filename>` 选择（auto 表示首次运行时自动下载该后端默认模型，
  下载到 ~/.cache/crispasr）；
- VAD 方法使用 `--vad -vm <keyword>` 选择。

精选已核对注册表及语言范围的 ASR 后端（不含纯 TTS / 辅助行），
另含 m2m100 / madlad 等文本翻译后端。
"""

CRISP_ASR_PROJECT_URL = "https://github.com/CrispStrobe/CrispASR"
CRISP_ASR_REPO = "CrispStrobe/CrispASR"


# Explicit downloads preserve the selected Whisper variant rather than falling
# back to the registry's default (ggml-base.bin).
CRISP_ASR_WHISPER_HF_REPOS = {
    **{f"ggml-{size}.bin": "ggerganov/whisper.cpp"
       for size in ("tiny", "base", "small", "medium", "large-v3")},
    "kotoba-whisper-v2.2-ggml-q8_0.bin": "kenrouse/kotoba-whisper-v2.2-ggml",
}


# VAD 方法关键字（传给 -vm/--vad-model）
CRISP_ASR_VAD_METHODS = [
    {"label": "Silero (默认, 标准, ~885KB)", "value": "silero"},
    {"label": "FireRedVAD (推荐, F1=97.57%, 2.4MB)", "value": "firered"},
    {"label": "MarbleNet (最小, 439KB, 6 语言)", "value": "marblenet"},
    {"label": "Whisper-VAD (实验性, 22MB, 较慢)", "value": "whisper-vad"},
]


# 音频语种识别（audio-LID）预处理模型，传给 --lid-backend。
# 用于没有原生语种识别能力的后端（见 _NO_NATIVE_LID_BACKENDS）在选择“自动检测”
# 时，由 CrispASR 先用该模型识别语言，再把结果传给主 ASR 后端。
CRISP_ASR_LID_METHODS = [
    {"label": "ECAPA (推荐, 107/45 语言, 40-43MB)", "value": "ecapa"},
    {"label": "Whisper-tiny (默认, ~99 语言, 75MB)", "value": "whisper"},
    {"label": "FireRed (120 语言, 544MB)", "value": "firered"},
    {"label": "Silero (95 语言, 16MB, 最小)", "value": "silero"},
]


# 表示“不限制下拉列表”的哨兵值：该后端/模型支持大规模/完整语言集，语言下拉
# 应展示完整枚举，不做筛选。是否可自动检测由 native-LID capability 另行判断。
_MULTI_50 = None


def _auto(
    label: str, backend: str, value: str = "auto", languages: "list[str] | None" = None
) -> dict:
    d = {"label": label, "value": value, "backend": backend}
    if languages is not _MULTI_50:
        d["languages"] = languages
    return d


# 后端引擎目录（全部 ASR 后端）。
# 同一“识别架构”家族（如 Parakeet）下，多个变体合并到同一后端项的模型下拉中；
# 其余每个后端单独成项，模型项默认“自动下载”。
#
# 每个后端字典可携带 "languages" 键：ISO 639-1 代码列表，表示该后端/模型实际支持
# 的语言子集；缺省或值为 None 表示不限制（保留完整语言下拉，支持自动检测/大语种集）。
# 模型字典也可携带自己的 "languages"，覆盖所属后端的默认值（如 Parakeet 内部各变体
# 语言差异很大）。用 get_language_codes() 解析最终生效的语言集合。
CRISP_ASR_BACKENDS = [
    {
        # 推荐：非自回归, 快且稳定, 适合长音频 (CPU 也流畅), 50+ 语言含日语
        "label": "SenseVoice (✨推荐✨, 50+ 语言, 快, 稳定)",
        "backend": "sensevoice",
        "languages": _MULTI_50,
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
        "languages": ["zh", "en"],
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
        "languages": _MULTI_50,
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
                # 由本模块 CRISP_ASR_WHISPER_HF_REPOS 映射处理。
                "label": "kotoba-whisper-v2.2 ✨ (日语, ggml q8_0)",
                "value": "kotoba-whisper-v2.2-ggml-q8_0.bin",
                "backend": "whisper",
                "languages": ["ja"],
            },
        ],
    },
    {
        # TDT/RNNT/混合变体使用 parakeet；独立 CTC 模型使用 fastconformer-ctc。
        # 具体变体通过 -m <registry-key> 选择（如 -m parakeet-ja），
        # 不要将仅用于模型注册表的 key 当作 --backend。
        "label": "Parakeet (NeMo, 快, 多语言, 词级时间戳)",
        "backend": "parakeet",
        "languages": _MULTI_50,
        "models": [
            _auto("TDT 0.6B v3 (多语言, 默认, 自动下载)", "parakeet", "auto", _MULTI_50),
            _auto("TDT 0.6B v2 (英语)", "parakeet", "parakeet-v2", ["en"]),
            _auto("TDT 0.6B (日语)", "parakeet", "parakeet-ja", ["ja"]),
            _auto("TDT 1.1B (英语, 更大)", "parakeet", "parakeet-tdt-1.1b", ["en"]),
            _auto("TDT-CTC 110M (英语, 最小)", "parakeet", "parakeet-tdt_ctc-110m", ["en"]),
            _auto("TDT-CTC 1.1B (含标点, 多语言)", "parakeet", "parakeet-tdt_ctc-1.1b", _MULTI_50),
            _auto("RNNT 0.6B (英语)", "parakeet", "parakeet-rnnt-0.6b", ["en"]),
            _auto("RNNT 1.1B (英语)", "parakeet", "parakeet-rnnt-1.1b", ["en"]),
            _auto("CTC 0.6B (英语)", "fastconformer-ctc", "parakeet-ctc-0.6b", ["en"]),
            _auto("CTC 1.1B (英语)", "fastconformer-ctc", "parakeet-ctc-1.1b", ["en"]),
            _auto("CTC 1.1B 日语 GAL (Q8, ~1.2GB)", "fastconformer-ctc", "parakeet-ctc-1.1b-ja", ["ja"]),
            _auto("ReazonSpeech NeMo v2 (日语, RNNT, Q8, ~704MB)", "parakeet", "reazonspeech", ["ja"]),
        ],
    },
    {
        "label": "FastConformer-CTC (NeMo, 英语)",
        "backend": "fastconformer-ctc",
        "languages": ["en"],
        "models": [_auto("FastConformer-CTC Large (自动下载)", "fastconformer-ctc")],
    },
    {
        "label": "Canary (NVIDIA, 25 欧语, 含翻译)",
        "backend": "canary",
        "languages": [
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
            "cs", "sk", "ro", "hu", "bg", "hr", "sl", "sv", "da", "fi",
            "el", "lt", "lv", "et", "mt",
        ],
        "models": [_auto("Canary 1B v2 (自动下载)", "canary")],
    },
    {
        "label": "Voxtral Mini 3B (Mistral, 8 语言)",
        "backend": "voxtral",
        "languages": ["en", "fr", "de", "es", "it", "pt", "nl", "hi"],
        "models": [
            _auto("Voxtral Mini 3B (自动下载)", "voxtral"),
            _auto("Voxtral Mini 3B Q8 (~4.6GB)", "voxtral", "voxtral-mini-3b-2507-q8_0.gguf"),
            _auto("Voxtral Mini 3B F16 (~8.7GB)", "voxtral", "voxtral-mini-3b-2507-f16.gguf"),
        ],
    },
    {
        "label": "Voxtral 4B Realtime (13 语言, 流式)",
        "backend": "voxtral4b",
        "languages": [
            "en", "fr", "de", "es", "it", "pt", "nl", "hi", "ar", "zh", "ja", "ko", "ru",
        ],
        "models": [
            _auto("Voxtral Mini 4B Realtime (自动下载)", "voxtral4b"),
            _auto("Voxtral 4B Realtime Q8 (~4.4GB)", "voxtral4b", "voxtral-mini-4b-realtime-q8_0.gguf"),
            _auto("Voxtral 4B Realtime F16 (~8.3GB)", "voxtral4b", "voxtral-mini-4b-realtime-f16.gguf"),
        ],
    },
    {
        "label": "Granite Speech (IBM)",
        "backend": "granite",
        "languages": ["en"],
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
        "languages": _MULTI_50,
        "models": [
            _auto("Qwen3-ASR 0.6B (自动下载)", "qwen3"),
            _auto("Qwen3-ASR 1.7B", "qwen3-1.7b"),
            _auto("Qwen3 1.7B 日语动漫 (Q4, ~1.3GB)", "qwen3", "qwen3-ja-anime", ["ja"]),
        ],
    },
    {
        "label": "Mega-ASR (Qwen3-1.7B, 抗噪)",
        "backend": "mega-asr",
        "languages": _MULTI_50,
        "models": [_auto("Mega-ASR 1.7B (自动下载)", "mega-asr")],
    },
    {
        "label": "Fun-ASR Nano (中粤英日韩)",
        "backend": "funasr",
        "languages": ["zh", "yue", "en", "ja", "ko"],
        "models": [
            _auto("Fun-ASR Nano (自动下载)", "funasr"),
            _auto("Fun-ASR MLT Nano (31 语言)", "fun-asr-mlt-nano", languages=[
                "zh", "yue", "en", "ja", "ko", "vi", "id", "th", "ms", "tl", "ar", "hi",
                "bg", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "ro", "el", "fi",
                "sv", "tr", "fa", "da", "hu", "mk", "ru",
            ]),
        ],
    },
    {
        "label": "Cohere Transcribe (13 语言)",
        "backend": "cohere",
        "languages": [
            "en", "fr", "de", "es", "it", "pt", "nl", "ja", "ko", "zh", "ar", "ru", "hi",
        ],
        "models": [
            _auto("Cohere Transcribe (自动下载)", "cohere"),
            _auto("Cohere Arabic (阿拉伯语, Q4 imatrix, ~1.5GB)", "cohere", "cohere-ar", ["ar"]),
            {
                "label": "Cohere ASR 日语",
                "value": "cohere-asr-ja-q4_k.gguf",
                "backend": "cohere",
                "languages": ["ja"],
            },
        ],
    },
    {
        "label": "wav2vec2 / HuBERT / data2vec (自监督 CTC)",
        "backend": "wav2vec2",
        "languages": ["en"],
        "models": [
            _auto("wav2vec2 XLSR (英语, 自动下载)", "wav2vec2"),
            _auto("wav2vec2 XLSR-53 (德语)", "wav2vec2", "wav2vec2-de", languages=["de"]),
            _auto("HuBERT Large (英语)", "hubert"),
            _auto("data2vec Audio (英语)", "data2vec"),
        ],
    },
    {
        "label": "omniASR (1600+ 语言)",
        "backend": "omniasr",
        "languages": _MULTI_50,
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
        "languages": ["zh"],
        "models": [_auto("FireRedASR2 AED (自动下载)", "firered-asr")],
    },
    {
        "label": "GLM-ASR Nano (17 语言, 含粤语)",
        "backend": "glm-asr",
        "languages": [
            "zh", "yue", "en", "ja", "ko", "fr", "de", "es", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "id", "tr",
        ],
        "models": [_auto("GLM-ASR Nano (自动下载)", "glm-asr")],
    },
    {
        "label": "Kyutai STT (英/法)",
        "backend": "kyutai-stt",
        "languages": ["en", "fr"],
        "models": [
            _auto("Kyutai STT 1B (自动下载)", "kyutai-stt"),
            _auto("Kyutai STT 2.6B (英语, ~1.5GB)", "kyutai-stt", "kyutai-stt-2.6b", ["en"]),
        ],
    },
    {
        "label": "Gemma4-E2B (140+ 语言, 需 HF 令牌)",
        "backend": "gemma4-e2b",
        "languages": _MULTI_50,
        "models": [_auto("Gemma4-E2B-IT (自动下载)", "gemma4-e2b")],
    },
    {
        "label": "MiMo-ASR (小米, 普通话 + 方言)",
        "backend": "mimo-asr",
        "languages": ["zh"],
        "models": [_auto("MiMo-ASR (自动下载, ~4.2GB)", "mimo-asr")],
    },
    {
        "label": "MOSS-Audio 4B (中英, 音频理解)",
        "backend": "moss-audio",
        "languages": ["zh", "en"],
        "models": [_auto("MOSS-Audio 4B Instruct (自动下载)", "moss-audio")],
    },
    {
        "label": "VibeVoice ASR (50+ 语言)",
        "backend": "vibevoice",
        "languages": _MULTI_50,
        "models": [_auto("VibeVoice ASR (自动下载, ~4.5GB)", "vibevoice")],
    },
    {
        "label": "GigaAM v3 (俄语, MIT)", "backend": "gigaam", "languages": ["ru"],
        "models": [_auto("GigaAM v3 e2e RNNT (Q8, ~249MB)", "gigaam")],
    },
    {
        "label": "Canary-Qwen 2.5B (英语, CC-BY-4.0)",
        "backend": "canary-qwen", "languages": ["en"],
        "models": [_auto("Canary-Qwen 2.5B (Q8, ~4.1GB, 须署名)", "canary-qwen")],
    },
    {
        "label": "Higgs-STT (英语, Apache-2.0)", "backend": "higgs-stt", "languages": ["en"],
        "models": [_auto("Higgs-STT (Q4, ~2.3GB)", "higgs-stt")],
    },
    {
        "label": "MOSS-Transcribe 2B (英语, Apache-2.0)",
        "backend": "moss-transcribe", "languages": ["en"],
        "models": [_auto("MOSS-Transcribe preview 2B (Q4, ~1.6GB)", "moss-transcribe")],
    },
    {
        "label": "Moonshine (轻量, 英语 + 6 语言)",
        "backend": "moonshine",
        "languages": ["en"],
        "models": [
            _auto("Moonshine Tiny (自动下载)", "moonshine"),
            _auto("Moonshine Base 德语", "moonshine-de", languages=["de"]),
            _auto("Moonshine Tiny 德语", "moonshine-tiny-de", languages=["de"]),
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


def model_id(model: dict) -> str:
    return f"{model['backend']}::{model['value']}"


def backend_options():
    return [(b["backend"], b["label"]) for b in CRISP_ASR_BACKENDS]


def model_options(backend):
    group = get_backend_by_label(backend)
    return [(model_id(m), m["label"]) for m in group["models"]] if group else []


def normalize_selection(backend, model, vad, lid):
    """Migrate recognized legacy labels only; unknown nonempty values survive."""
    group = get_backend_by_label(backend)
    if group:
        backend = group["backend"]
        for entry in group["models"]:
            if model in (entry["label"], model_id(entry)):
                model = model_id(entry)
                break
    for entries, value, name in ((CRISP_ASR_VAD_METHODS, vad, "vad"), (CRISP_ASR_LID_METHODS, lid, "lid")):
        for entry in entries:
            if value in (entry["label"], entry["value"]):
                if name == "vad":
                    vad = entry["value"]
                else:
                    lid = entry["value"]
                break
    return backend, model, vad, lid


def get_backend_labels() -> list[str]:
    return [b["label"] for b in CRISP_ASR_BACKENDS]


def get_backend_by_label(label: str) -> "dict | None":
    for b in CRISP_ASR_BACKENDS:
        if label in (b["label"], b["backend"]):
            return b
    return None


def get_model_labels(backend_label: str) -> list[str]:
    b = get_backend_by_label(backend_label)
    if not b:
        return []
    return [m["label"] for m in b["models"]]


def get_language_codes(backend_label: str, model_label: "str | None" = None) -> "list[str] | None":
    """返回给定后端（可选模型）实际支持的 ISO 639-1 语言代码列表。

    返回 None 表示不限制（该后端/模型支持自动检测 + 大规模/完整语言集，
    语言下拉应展示完整枚举）。解析优先级：模型级 "languages" 覆盖 > 后端级
    "languages" 默认值 > 不限制（None）。
    """
    b = get_backend_by_label(backend_label)
    if not b:
        return None
    if model_label is not None:
        for m in b["models"]:
            if model_label in (m["label"], model_id(m)):
                if "languages" in m:
                    return m["languages"]
                break
    return b.get("languages")


# CrispASR v0.8.34 的 --list-backends-json 运行时 capability matrix 中，只有
# Whisper 与 SenseVoice 声明 language-detect。其余 ASR 后端的 `-l auto` 都会走
# 外部 audio-LID 预处理步骤，默认会使用 whisper-tiny；本应用允许用户改选 ECAPA、
# FireRed 等 LID 模型。不要以 README 的语言覆盖表推断原生 LID。
_NATIVE_LID_BACKENDS = {"whisper", "sensevoice"}


def _resolve_effective_backend(backend_label: str, model_label: "str | None" = None) -> "str | None":
    """解析给定后端标签 + 模型标签实际生效的 --backend 值（模型可覆盖所属后端）。"""
    b = get_backend_by_label(backend_label)
    if not b:
        return None
    effective_backend = b["backend"]
    if model_label is not None:
        for m in b["models"]:
            if model_label in (m["label"], model_id(m)):
                effective_backend = m.get("backend", b["backend"])
                break
    return effective_backend


def supports_auto_lid(backend_label: str, model_label: "str | None" = None) -> bool:
    """判断给定后端（可选模型）是否具备原生语种自动识别（native audio-LID）能力。

    条件：该组合的语言集不受限（get_language_codes 返回 None，即支持多语言/大语种
    集），且实际生效的 --backend 值在 _NATIVE_LID_BACKENDS 名单中。
    """
    if get_language_codes(backend_label, model_label) is not None:
        return False
    effective_backend = _resolve_effective_backend(backend_label, model_label)
    if effective_backend is None:
        return False
    return effective_backend in _NATIVE_LID_BACKENDS


def needs_lid_pre_step(backend_label: str, model_label: "str | None" = None) -> bool:
    """判断给定后端（可选模型）是否需要靠额外 LID 预处理步骤才能提供“自动检测”。

    条件：实际生效的 --backend 值不在 _NATIVE_LID_BACKENDS 名单中，且该组合支持
    2 种及以上语言（只支持 1 种语言时谈不上“检测”，不需要也不应展示自动检测项）。
    """
    effective_backend = _resolve_effective_backend(backend_label, model_label)
    if effective_backend is None or effective_backend in _NATIVE_LID_BACKENDS:
        return False
    codes = get_language_codes(backend_label, model_label)
    return codes is None or len(codes) >= 2


def get_lid_labels() -> list[str]:
    return [m["label"] for m in CRISP_ASR_LID_METHODS]


def resolve_lid_method(lid_label: str) -> str:
    for m in CRISP_ASR_LID_METHODS:
        if lid_label in (m["label"], m["value"]):
            return m["value"]
    if lid_label:
        raise ValueError(f"未知 LID 选择: {lid_label}")
    return "ecapa"


def resolve_model(backend_label: str, model_label: str) -> "tuple[str, str]":
    """根据后端标签与模型标签，返回 (实际 --backend 值, -m 值)。

    模型项可携带自己的 backend（如 Parakeet 的多个变体对应不同 --backend）。
    """
    if backend_label.startswith("KugelAudio"):
        raise ValueError("KugelAudio 仅支持语音合成，不能转录。请在设置中重新选择 ASR 模型。")
    b = get_backend_by_label(backend_label)
    if not b:
        if backend_label or model_label:
            raise ValueError(f"未知 ASR 选择: {backend_label}；请重新选择模型")
        return ("whisper", "auto")
    for m in b["models"]:
        if model_label in (m["label"], model_id(m)):
            return (m.get("backend", b["backend"]), m["value"])
    if model_label:
        raise ValueError(f"未知模型选择: {model_label}；不会自动改用其他模型")
    if b["models"]:
        first = b["models"][0]
        return (first.get("backend", b["backend"]), first["value"])
    return (b["backend"], "auto")


def get_vad_labels() -> list[str]:
    return [v["label"] for v in CRISP_ASR_VAD_METHODS]


def resolve_vad_method(vad_label: str) -> str:
    for v in CRISP_ASR_VAD_METHODS:
        if vad_label in (v["label"], v["value"]):
            return v["value"]
    if vad_label:
        raise ValueError(f"未知 VAD 选择: {vad_label}")
    return "silero"
