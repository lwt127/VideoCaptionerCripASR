import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...config import BIN_PATH, MODEL_PATH
from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

logger = setup_logger("crisp_asr")

# CrispASR 二进制目录（随应用打包在 resource/bin/CrispASR 下）
CRISP_ASR_BIN = Path(BIN_PATH) / "CrispASR" / "crispasr.exe"

# 非内置注册表的 whisper ggml 模型 → HuggingFace 仓库映射。
# 这些模型需通过 `--hf-repo OWNER/REPO --hf-file FILE` 从自定义仓库下载，
# CrispASR 会缓存到 ~/.cache/crispasr。键为 -m 传入的文件名（小写匹配）。
CRISP_ASR_WHISPER_HF_REPOS = {
    "kotoba-whisper-v2.2-ggml-q8_0.bin": "kenrouse/kotoba-whisper-v2.2-ggml",
    "kotoba-whisper-v2.2-ggml-q5_0.bin": "kenrouse/kotoba-whisper-v2.2-ggml",
    "kotoba-whisper-v2.2-ggml.bin": "kenrouse/kotoba-whisper-v2.2-ggml",
}


def cuda_available() -> bool:
    """检测系统是否有可用的 NVIDIA GPU（通过 nvidia-smi）。"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def engine_supports_cuda(exe_path) -> bool:
    """运行 crispasr --version，判断该二进制的 ggml 后端是否包含 cuda/vulkan(GPU)。"""
    try:
        r = subprocess.run(
            [str(exe_path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        out = (r.stdout or "") + (r.stderr or "")
        # 形如 "ggml backends : cpu" 或 "... cuda ..."
        for line in out.splitlines():
            if "backends" in line.lower():
                low = line.lower()
                return "cuda" in low or "vulkan" in low
        return False
    except Exception:
        return False


class CrispASR(BaseASR):
    """CrispASR 本地转录后端（多后端 ASR 引擎）。

    CrispASR 是 whisper.cpp 的兼容分支，支持多种识别后端（whisper / parakeet /
    funasr / sensevoice / voxtral 等），输出标准 SRT 字幕。

    - 后端通过 ``--backend <name>`` 选择；
    - 模型通过 ``-m <auto|filename>`` 选择（auto 首次运行时自动下载）；
    - VAD 通过 ``--vad -vm <method>`` 选择；
    - 默认启用 GPU，可通过 ``--no-gpu`` 关闭。

    用法示例:
        crispasr.exe --backend parakeet -m auto -f <audio.wav> -l <lang> \
            -osrt -of <base> --vad -vm silero
    """

    # 单个 VAD 语音区间的最大时长（秒）。CrispASR 默认不限制（FLT_MAX），
    # 会导致一段连续语音（尤其无标点/无停顿）变成 20~30 秒的超长字幕。
    # 超过该时长的语音区间会被自动切分。
    VAD_MAX_SPEECH_DURATION_S = 8

    def __init__(
        self,
        audio_path,
        language="en",
        crisp_asr_path=None,
        backend="whisper",
        model="auto",
        vad_method="silero",
        use_gpu: bool = True,
        use_vad: bool = True,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_path, False)
        assert os.path.exists(audio_path), f"音频文件 {audio_path} 不存在"
        assert audio_path.endswith(".wav"), f"音频文件 {audio_path} 必须是WAV格式"

        self.backend = backend or "whisper"
        self.vad_method = vad_method or "silero"
        # 若该 whisper ggml 模型需要从自定义 HF 仓库下载，则记录其仓库；否则为 None。
        self.hf_repo: "str | None" = None

        # 解析模型参数：
        #  - "auto" → 交由 CrispASR 自动下载该后端默认模型（缓存到 ~/.cache/crispasr）
        #  - "ggml-*.bin"（whisper 后端）→ 优先用本地 models 目录的文件；
        #    若本地不存在，则回退为 "auto"，让 CrispASR 自动下载默认 whisper 模型。
        if model and model != "auto" and "ggml" in model.lower():
            models_dir = Path(MODEL_PATH)
            candidate = models_dir / model
            hf_repo = CRISP_ASR_WHISPER_HF_REPOS.get(model.lower())
            if candidate.exists():
                self.model_arg = str(candidate)
                logger.info(f"使用本地模型文件: {self.model_arg}")
            else:
                matches = list(models_dir.glob(f"*{Path(model).stem}*.bin"))
                if matches:
                    self.model_arg = str(matches[0])
                    logger.info(f"使用本地模型文件: {self.model_arg}")
                elif hf_repo:
                    # 自定义仓库的 ggml 模型（如 kotoba-whisper）：保留文件名，
                    # 通过 --hf-repo 从 HuggingFace 下载（CrispASR 缓存到 ~/.cache/crispasr）。
                    self.model_arg = model
                    self.hf_repo = hf_repo
                    logger.info(
                        f"使用 HuggingFace 仓库模型: --hf-repo {hf_repo} --hf-file {model}"
                    )
                else:
                    # 本地没有该 ggml 模型 → 交给 CrispASR 自动下载
                    self.model_arg = "auto"
                    logger.info(
                        f"本地未找到模型 '{model}'，改用自动下载 (backend={self.backend})"
                    )
        else:
            # 自动下载（CrispASR 会缓存到 ~/.cache/crispasr）
            self.model_arg = model or "auto"
            logger.info(
                f"使用自动下载模型: backend={self.backend}, model={self.model_arg}"
            )

        # 定位 crispasr 可执行文件（缺失时在 _run 阶段自动下载，不在此处抛错）
        self.crisp_asr_path = Path(crisp_asr_path) if crisp_asr_path else CRISP_ASR_BIN

        self.language = language
        self.use_gpu = use_gpu
        self.use_vad = use_vad
        self.need_word_time_stamp = need_word_time_stamp
        self.process = None

    def _ensure_engine(self, callback) -> None:
        """确保 crispasr 引擎二进制存在；缺失则从 GitHub Releases 自动下载。

        在转录线程内同步执行，进度通过 callback 反馈，不阻塞 UI 线程。

        若开启 GPU 且检测到 NVIDIA 显卡，但当前引擎为 CPU-only 构建，
        则自动下载 CUDA 构建以启用 GPU 加速。
        """
        want_cuda = self.use_gpu and cuda_available()

        # 已存在引擎时：若需要 CUDA 但当前为 CPU-only，则升级为 CUDA 构建
        if self.crisp_asr_path.exists():
            if want_cuda and not engine_supports_cuda(self.crisp_asr_path):
                logger.info("检测到 NVIDIA GPU，但当前 CrispASR 为 CPU 构建，升级为 CUDA 构建…")
                callback(0, "检测到显卡，正在下载 CUDA 版 CrispASR 引擎…")
                self._download_engine(callback, prefer_gpu=True)
            else:
                if want_cuda:
                    logger.info("CrispASR 引擎已支持 CUDA，使用 GPU 加速")
            return

        logger.info("未检测到 CrispASR 引擎，开始自动下载…")
        callback(0, "正在下载 CrispASR 引擎…")
        self._download_engine(callback, prefer_gpu=want_cuda)

    def _download_engine(self, callback, prefer_gpu: bool):
        """下载并安装 CrispASR 引擎（CPU 或 CUDA 构建）。"""
        try:
            from app.thread.crisp_asr_download_thread import (
                download_crisp_asr_engine_sync,
            )
        except Exception as e:  # pragma: no cover
            raise FileNotFoundError(
                f"未找到 CrispASR 引擎，且无法加载下载模块: {e}"
            )

        target_dir = Path(CRISP_ASR_BIN).parent

        def _dl_progress(pct: int, msg: str):
            # 引擎下载占进度前 30%，避免与转录进度冲突
            callback(min(int(pct * 0.3), 30), msg)

        exe = download_crisp_asr_engine_sync(
            target_dir,
            prefer_gpu=prefer_gpu,
            progress=_dl_progress,
            force=True,  # 允许覆盖已有（CPU→CUDA 升级）
        )
        self.crisp_asr_path = Path(exe)
        logger.info("CrispASR 引擎已就绪: %s (GPU=%s)", self.crisp_asr_path, prefer_gpu)
        callback(30, "引擎就绪，开始转录…")

    def _make_segments(self, resp_data: str) -> list[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        # 过滤掉纯音乐/音效标记
        filtered_segments = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            if not (
                text.startswith("【")
                or text.startswith("[")
                or text.startswith("(")
                or text.startswith("（")
            ):
                filtered_segments.append(seg)

        # Python 侧后处理：拆分过长字幕段（后端无关）。
        # 某些后端/VAD 组合（如 cohere + whisper-vad）不响应
        # --vad-max-speech-duration-s，会产出 20~30 秒的超长字幕。
        # 这里把超过阈值的段在标点处切分，时间按字符比例分配，
        # 保证任何后端都能得到长度可控的字幕。
        split_segments: list[ASRDataSeg] = []
        for seg in filtered_segments:
            split_segments.extend(self._split_long_segment(seg))
        return split_segments

    def _split_long_segment(self, seg: ASRDataSeg) -> list[ASRDataSeg]:
        """将时长超过阈值的字幕段递归拆分为更短的段。

        - 优先在句末标点（。！？!?.）后切分；
        - 其次在次级标点（、，,…）或空格处切分；
        - 实在无标点可切时，按字符中点强制切分；
        - 子段时间按各自字符数占比在原段时间区间内线性分配。
        """
        duration_ms = seg.end_time - seg.start_time
        text = seg.text.strip()
        max_ms = int(self.VAD_MAX_SPEECH_DURATION_S * 1000)

        # 足够短，或没有可分文本，直接返回
        if duration_ms <= max_ms or len(text) <= 1:
            return [seg]

        # 选择切分位置：尽量靠近文本中点的标点
        split_idx = self._find_split_index(text)
        if split_idx is None:
            # 没有任何标点，按字符中点硬切
            split_idx = len(text) // 2

        left_text = text[:split_idx].strip()
        right_text = text[split_idx:].strip()
        if not left_text or not right_text:
            # 切分无效（例如标点在首尾），返回原段避免死循环
            return [seg]

        # 按字符数占比分配时间
        total_chars = len(left_text) + len(right_text)
        left_ratio = len(left_text) / total_chars
        mid_time = seg.start_time + int(duration_ms * left_ratio)

        left_seg = ASRDataSeg(left_text, seg.start_time, mid_time)
        right_seg = ASRDataSeg(right_text, mid_time, seg.end_time)

        # 递归处理，直到所有子段都不超过阈值
        result: list[ASRDataSeg] = []
        result.extend(self._split_long_segment(left_seg))
        result.extend(self._split_long_segment(right_seg))
        return result

    @staticmethod
    def _find_split_index(text: str) -> "int | None":
        """返回最接近文本中点的切分位置（标点后一位）。

        先找句末标点，找不到再找次级标点/空格。返回 None 表示无标点。
        """
        primary = "。！？!?."
        secondary = "、，,…　 "
        mid = len(text) / 2

        def best_after(chars: str) -> "int | None":
            positions = [i + 1 for i, c in enumerate(text) if c in chars]
            # 排除位于首尾、会切出空串的位置
            positions = [p for p in positions if 0 < p < len(text)]
            if not positions:
                return None
            # 选离中点最近的切分点，使左右更均衡
            return min(positions, key=lambda p: abs(p - mid))

        return best_after(primary) or best_after(secondary)

    def _build_command(self, wav_path: Path, output_base: Path) -> list[str]:
        """构建 crispasr 命令行参数。

        Args:
            wav_path: 输入 WAV 文件路径
            output_base: 输出文件基础路径（不含扩展名，CrispASR 会追加 .srt）
        """
        params = [
            str(self.crisp_asr_path),
            "--backend",
            self.backend,
            "-m",
            str(self.model_arg),
            "-f",
            str(wav_path),
            "-l",
            self.language,
            "--output-srt",
            "--output-file",
            str(output_base),
        ]

        # 自定义 HuggingFace 仓库模型（如 kotoba-whisper）：通过 --hf-repo / --hf-file
        # 让 CrispASR 从指定仓库下载并缓存（--hf-repo 本身即隐含 auto-download）。
        if self.hf_repo:
            params.extend(["--hf-repo", self.hf_repo, "--hf-file", str(self.model_arg)])

        # 当指定了具体模型文件名（而非本地已存在的路径，也非 "auto"）时，
        # 该模型可能尚未下载到 CrispASR 缓存（~/.cache/crispasr）。
        # 此时显式加上 --auto-download，让引擎按名称自动下载该模型，
        # 否则会以 "model not found locally" 报错退出 (code 13)。
        # 说明：
        #  - model_arg == "auto" 时引擎本身即会自动下载，无需此参数；
        #  - model_arg 为本地已存在的绝对路径时，os.path.exists 为真，跳过；
        #  - 使用 --hf-repo 时已隐含自动下载，无需重复追加；
        #  - 其余情况（如 cohere-asr-ja-q4_k.gguf 这类按名引用的模型）才追加。
        elif self.model_arg != "auto" and not os.path.exists(str(self.model_arg)):
            params.append("--auto-download")

        # 在句末标点（. ! ? 。！？）处断句，生成更适合字幕的短句。
        params.append("--split-on-punct")

        # GPU 控制：默认启用 GPU，关闭时传 --no-gpu
        if not self.use_gpu:
            params.append("--no-gpu")

        # VAD 分段（更适合字幕场景），并指定 VAD 方法
        if self.use_vad:
            params.append("--vad")
            if self.vad_method and self.vad_method != "silero":
                params.extend(["--vad-model", self.vad_method])
            # 关键：限制单个语音区间的最大时长（默认 FLT_MAX 表示不限制，
            # 正是出现 20~30 秒超长字幕的根因）。超过该时长的语音会被自动切分。
            params.extend(
                ["--vad-max-speech-duration-s", str(self.VAD_MAX_SPEECH_DURATION_S)]
            )

        # 中文模式下添加提示语（whisper 后端支持 --prompt）
        if self.language == "zh" and self.backend == "whisper":
            params.extend(
                ["--prompt", "你好，我们需要使用简体中文，以下是普通话的句子。"]
            )

        return params

    def _run(self, callback=None) -> str:
        if callback is None:
            callback = lambda x, y: None

        # 确保引擎可用：缺失则自动下载（模型则由 CrispASR 自身按需自动下载）
        self._ensure_engine(callback)

        temp_root = Path(tempfile.gettempdir()) / "bk_asr"
        temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=temp_root) as temp_path:
            temp_dir = Path(temp_path)
            wav_path = temp_dir / "audio.wav"
            output_base = temp_dir / "audio"
            output_srt = output_base.with_suffix(".srt")

            try:
                shutil.copy2(self.audio_path, wav_path)

                params = self._build_command(wav_path, output_base)
                logger.info("完整命令行参数: %s", " ".join(params))

                self.process = subprocess.Popen(
                    params,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    ),
                )

                total_duration = self.get_audio_duration(self.audio_path) or 600
                logger.info("音频总时长: %d 秒", total_duration)

                full_output = []
                while True:
                    try:
                        line = self.process.stdout.readline()
                    except Exception:
                        break
                    if not line:
                        if self.process.poll() is not None:
                            break
                        continue

                    full_output.append(line)

                    # 实时输出 CrispASR 日志（去掉行尾换行）
                    stripped = line.rstrip("\r\n")
                    if stripped:
                        logger.info("[CrispASR] %s", stripped)

                    # 解析进度（whisper.cpp 兼容输出: [HH:MM:SS.mmm --> ...]）
                    if " --> " in line and "[" in line:
                        try:
                            time_str = line.split("[")[1].split(" -->")[0].strip()
                            current_time = sum(
                                float(x) * y
                                for x, y in zip(
                                    reversed(time_str.split(":")), [1, 60, 3600]
                                )
                            )
                            progress = int(min(current_time / total_duration * 100, 98))
                            callback(progress, f"{progress}% 正在转换")
                        except (ValueError, IndexError):
                            continue

                self.process.wait()

                if self.process.returncode != 0:
                    code = self.process.returncode
                    # 0xFFFFFFFF / -1 等通常为后端崩溃（如长音频 + 实验性后端不稳定）
                    hint = ""
                    if code in (4294967295, -1, 3221225477, -1073741819):
                        hint = (
                            "（该后端在当前音频上崩溃，建议改用更稳定的后端"
                            "如 SenseVoice / Paraformer-zh / Whisper，"
                            "或将 VAD 方法改为 Silero）"
                        )
                    raise RuntimeError(
                        f"CrispASR 执行失败 (code {code}){hint}: "
                        + "".join(full_output[-20:])
                    )

                callback(100, "转换完成")

                if not output_srt.exists():
                    raise RuntimeError(f"输出文件未生成: {output_srt}")

                # 某些后端在长音频上可能输出被截断的多字节字符，导致严格 UTF-8 解码失败。
                # 使用 errors="replace" 容错读取，避免整个转录结果因个别坏字节而丢失。
                return output_srt.read_text(encoding="utf-8", errors="replace")

            except Exception as e:
                logger.exception("CrispASR 处理失败")
                raise RuntimeError(f"生成 SRT 文件失败: {str(e)}")

    def _get_key(self):
        # "sp" 标记启用了 --split-on-punct，确保与旧缓存（未断句结果）区分开
        return (
            f"crispasr-{self.crc32_hex}-{self.need_word_time_stamp}"
            f"-{self.backend}-{Path(str(self.model_arg)).name}-{self.language}-sp"
        )

    def get_audio_duration(self, filepath: str) -> int:
        try:
            cmd = ["ffmpeg", "-i", filepath]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            info = result.stderr
            if duration_match := re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info):
                hours, minutes, seconds = map(float, duration_match.groups())
                return int(hours * 3600 + minutes * 60 + seconds)
            return 600
        except Exception as e:
            logger.exception("获取音频时长时出错: %s", str(e))
            return 600
