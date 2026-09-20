"""Reviewed restrictions for the models exposed by our CrispASR catalog.

Acceptance is scoped to the resolved backend/model, license, and review revision.
Unknown future restrictions remain enforced by CrispASR; never pass "all".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLicense:
    tag: str
    url: str
    notice: str
    revision: str = "crispasr-0.8.34"


_FUNASR_NOTICE = "FunASR Model License v1.1：请阅读归属声明及使用条款；不等于禁止商业使用。"
MODEL_LICENSES = {
    "sensevoice": ModelLicense(
        "funasr-v1.1", "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE", _FUNASR_NOTICE
    ),
    "paraformer": ModelLicense(
        "funasr-v1.1", "https://huggingface.co/funasr/paraformer-zh", _FUNASR_NOTICE
    ),
    "funasr": ModelLicense(
        "funasr-v1.1", "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/blob/main/LICENSE", _FUNASR_NOTICE
    ),
    "fun-asr-mlt-nano": ModelLicense(
        "funasr-v1.1", "https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/LICENSE", _FUNASR_NOTICE
    ),
    "gemma4-e2b": ModelLicense(
        "gemma-terms", "https://ai.google.dev/gemma/terms", "请阅读并遵守 Gemma 使用条款及禁止使用政策。"
    ),
    "moonshine-de": ModelLicense(
        "cc-by-nc-sa-4.0", "https://huggingface.co/fidoriel/moonshine-base-de",
        "CC BY-NC-SA 4.0：仅限非商业用途，须署名并按相同方式共享。"
    ),
    "moonshine-tiny-de": ModelLicense(
        "cc-by-nc-sa-4.0", "https://huggingface.co/fidoriel/moonshine-tiny-de",
        "CC BY-NC-SA 4.0：仅限非商业用途，须署名并按相同方式共享。"
    ),
}


def model_license(backend: str, model: str = "auto") -> ModelLicense | None:
    # Registry variants passed as -m can override the backend's default model.
    return MODEL_LICENSES.get(model) or MODEL_LICENSES.get(backend)


def acceptance_key(backend: str, model: str, license: ModelLicense) -> str:
    return f"{license.revision}|{backend}|{model}|{license.tag}"


def accepted_license_tag(backend: str, model: str, acceptances) -> str | None:
    license = model_license(backend, model)
    if license is None:
        return None
    if isinstance(acceptances, (list, tuple)) and acceptance_key(backend, model, license) in acceptances:
        return license.tag
    raise ValueError(
        f"模型 {backend}/{model} 尚未确认授权 ({license.tag})。"
        "请在 CrispASR 设置中点击「查看并确认授权」，或在开始任务时确认。"
    )