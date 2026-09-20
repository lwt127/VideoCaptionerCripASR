"""Stable option IDs, display labels and validated v0.8.34 CLI contracts."""

ENGINE_FLAVORS = {
    "auto": "自动（NVIDIA → CUDA 12，否则 CPU）",
    "cuda": "CUDA 12（NVIDIA，兼容 Pascal/Volta）",
    "cuda13": "CUDA 13（Turing 或更新，驱动 580+）",
    "vulkan": "Vulkan（AMD / Intel / NVIDIA）",
    "cpu": "CPU（AVX2 / FMA / F16C）",
    "cpu-legacy": "Legacy CPU（旧 x86-64 / SSE2）",
}
ALIGNMENT_MODES = {
    "default": "引擎默认（保留原有行为）",
    "native": "仅原生时间戳（不下载对齐模型）",
    "fallback": "补充对齐（保留已有逐字时间戳）",
    "force": "强制重新对齐（使用所选对齐模型）",
}

# auto is deliberately not exposed: upstream auto is Canary, not language-aware.
ALIGNERS = {
    "qwen3-forced-aligner": {
        "label": "Qwen3 Forced Aligner（多语言，~500MB，Apache-2.0）",
        "languages": None,
    },
    "canary-ctc-aligner": {
        "label": "Canary CTC（25 欧语，~442MB，CC-BY-4.0 须署名）",
        "languages": ("en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
                      "cs", "sk", "ro", "hu", "bg", "hr", "sl", "sv", "da", "fi",
                      "el", "lt", "lv", "et", "mt"),
    },
    "fastconformer-aligner-en": {
        "label": "FastConformer 英语（~83MB，CC-BY-4.0 须署名）", "languages": ("en",),
    },
    **{
        f"wav2vec2-aligner-{code}": {
            "label": f"wav2vec2 {name}（~212–300MB，Apache-2.0）", "languages": (code,),
        }
        for code, name in (("ja", "日语"), ("zh", "中文"), ("en", "英语"), ("de", "德语"),
                           ("fr", "法语"), ("es", "西班牙语"), ("it", "意大利语"),
                           ("nl", "荷兰语"), ("pt", "葡萄牙语"), ("ar", "阿拉伯语"),
                           ("uk", "乌克兰语"), ("cs", "捷克语"))
    },
}


def alignment_args(mode: str, aligner: str, language: str, word_timestamps: bool) -> list[str]:
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"未知对齐模式: {mode}")
    args = []
    if mode == "native":
        args.append("--no-auto-aligner")
    elif mode in ("fallback", "force"):
        if aligner not in ALIGNERS:
            raise ValueError(f"未知对齐模型: {aligner}")
        languages = ALIGNERS[aligner]["languages"]
        if languages is not None and language not in languages:
            raise ValueError("对齐模型与源语言不匹配。请指定对应源语言，或选用多语言 Qwen3 对齐模型。")
        args.extend(["--aligner-model", aligner])
        if mode == "force":
            args.append("--force-aligner")  # NOT -fa (flash attention).
    if word_timestamps:
        # -ml 1 alone can interpolate timings. Refuse missing real word data.
        args.extend(["--max-len", "1", "--require-word-timestamps"])
    return args