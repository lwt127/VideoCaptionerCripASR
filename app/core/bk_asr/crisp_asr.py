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

        # 解析模型参数：
        #  - "auto" → 交由 CrispASR 自动下载该后端默认模型（缓存到 ~/.cache/crispasr）
        #  - "ggml-*.bin"（whisper 后端）→ 优先用本地 models 目录的文件；
        #    若本地不存在，则回退为 "auto"，让 CrispASR 自动下载默认 whisper 模型。
        if model and model != "auto" and "ggml" in model.lower():
            models_dir = Path(MODEL_PATH)
            candidate = models_dir / model
            if candidate.exists():
                self.model_arg = str(candidate)
                logger.info(f"使用本地模型文件: {self.model_arg}")
            else:
                matches = list(models_dir.glob(f"*{Path(model).stem}*.bin"))
                if matches:
                    self.model_arg = str(matches[0])
                    logger.info(f"使用本地模型文件: {self.model_arg}")
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
        return filtered_segments

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

        # GPU 控制：默认启用 GPU，关闭时传 --no-gpu
        if not self.use_gpu:
            params.append("--no-gpu")

        # VAD 分段（更适合字幕场景），并指定 VAD 方法
        if self.use_vad:
            params.append("--vad")
            if self.vad_method and self.vad_method != "silero":
                params.extend(["--vad-model", self.vad_method])

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

                return output_srt.read_text(encoding="utf-8")

            except Exception as e:
                logger.exception("CrispASR 处理失败")
                raise RuntimeError(f"生成 SRT 文件失败: {str(e)}")

    def _get_key(self):
        return (
            f"crispasr-{self.crc32_hex}-{self.need_word_time_stamp}"
            f"-{self.backend}-{Path(str(self.model_arg)).name}-{self.language}"
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
