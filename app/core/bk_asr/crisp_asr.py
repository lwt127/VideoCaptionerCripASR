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


class CrispASR(BaseASR):
    """CrispASR 本地转录后端。

    CrispASR 是 whisper.cpp 的兼容分支，CLI 参数为 whisper.cpp 的超集，
    输出标准 SRT 字幕，并复用 whisper.cpp 的 ggml-*.bin 模型文件。

    用法示例:
        crispasr.exe -m <model.bin> -f <audio.wav> -l <lang> -osrt -of <base>
    """

    def __init__(
        self,
        audio_path,
        language="en",
        crisp_asr_path=None,
        whisper_model=None,
        use_gpu: bool = False,
        use_vad: bool = True,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
    ):
        super().__init__(audio_path, False)
        assert os.path.exists(audio_path), f"音频文件 {audio_path} 不存在"
        assert audio_path.endswith(".wav"), f"音频文件 {audio_path} 必须是WAV格式"

        # 在 models 目录下查找对应的 ggml 模型（与 WhisperCpp 共用）
        if whisper_model:
            models_dir = Path(MODEL_PATH)
            model_files = list(models_dir.glob(f"*ggml*{whisper_model}*.bin"))
            if not model_files:
                raise ValueError(
                    f"在 {models_dir} 目录下未找到包含 '{whisper_model}' 的 ggml 模型文件。"
                    f"请在「转录」设置中下载 WhisperCpp 模型（CrispASR 复用相同模型）。"
                )
            model_path = str(model_files[0])
            logger.info(f"找到模型文件: {model_path}")
        else:
            raise ValueError("whisper_model 不能为空")

        # 定位 crispasr 可执行文件
        self.crisp_asr_path = Path(crisp_asr_path) if crisp_asr_path else CRISP_ASR_BIN
        if not self.crisp_asr_path.exists():
            raise FileNotFoundError(
                f"未找到 CrispASR 可执行文件: {self.crisp_asr_path}"
            )

        self.model_path = model_path
        self.language = language
        self.use_gpu = use_gpu
        self.use_vad = use_vad
        self.need_word_time_stamp = need_word_time_stamp
        self.process = None

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
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            "-l",
            self.language,
            "--output-srt",
            "--output-file",
            str(output_base),
        ]

        # GPU 控制：默认禁用 GPU（与 WhisperCpp 行为保持一致，避免无显卡环境报错）
        if not self.use_gpu:
            params.append("--no-gpu")

        # VAD 分段（更适合字幕场景）
        if self.use_vad:
            params.append("--vad")

        # 中文模式下添加提示语
        if self.language == "zh":
            params.extend(
                ["--prompt", "你好，我们需要使用简体中文，以下是普通话的句子。"]
            )

        return params

    def _run(self, callback=None) -> str:
        if callback is None:
            callback = lambda x, y: None

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
                    raise RuntimeError(
                        f"CrispASR 执行失败 (code {self.process.returncode}): "
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
            f"-{Path(self.model_path).name}-{self.language}"
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
