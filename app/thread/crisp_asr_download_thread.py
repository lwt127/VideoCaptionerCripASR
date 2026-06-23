"""CrispASR 引擎二进制自动下载线程。

如果用户仓库中缺少 crispasr 可执行文件，则从 GitHub Releases 自动下载
对应平台的预编译包并解压到 resource/bin/CrispASR/，实现开箱即用。

Windows 资产命名（来自 CrispASR release.yml）:
    crispasr-windows-x86_64-cpu.zip          (AVX2, 自包含, 推荐)
    crispasr-windows-x86_64-cpu-legacy.zip   (SSE4.2, 旧 CPU)
    crispasr-windows-x86_64-cuda.zip         (NVIDIA CUDA)
    crispasr-windows-x86_64-vulkan.zip       (Vulkan GPU)
"""

import json
import os
import platform
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.bk_asr.crisp_asr_catalog import CRISP_ASR_REPO
from app.core.utils.logger import setup_logger

logger = setup_logger("crisp_asr_download")


def _windows_asset_name(prefer_gpu: bool = False) -> str:
    """选择 Windows 平台的发行包资产名。默认使用自包含的 CPU 版（最稳）。"""
    if prefer_gpu:
        return "crispasr-windows-x86_64-cuda.zip"
    return "crispasr-windows-x86_64-cpu.zip"


def get_release_asset_url(prefer_gpu: bool = False) -> str:
    """查询 CrispASR 最新 Release，返回匹配当前平台的资产下载链接。"""
    system = platform.system()
    if system != "Windows":
        raise RuntimeError(
            f"暂仅支持 Windows 自动下载 CrispASR 引擎，请手动构建（当前系统: {system}）"
        )

    asset_name = _windows_asset_name(prefer_gpu)
    api_url = f"https://api.github.com/repos/{CRISP_ASR_REPO}/releases/latest"

    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        api_url, headers={"User-Agent": "VideoCaptioner-CrispASR"}
    )
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    assets = data.get("assets", [])
    # 优先精确匹配，否则回退到任意 windows 的 cpu 包
    for a in assets:
        if a.get("name") == asset_name:
            return a["browser_download_url"]
    for a in assets:
        name = a.get("name", "")
        if "windows" in name and "cpu" in name and name.endswith(".zip"):
            return a["browser_download_url"]
    raise RuntimeError(
        f"在最新 Release 中未找到 Windows 资产（期望: {asset_name}）"
    )


class CrispASRDownloadThread(QThread):
    """下载并解压 CrispASR 引擎二进制到目标目录。"""

    progress = pyqtSignal(int, str)  # (百分比, 状态文本)
    finished = pyqtSignal(str)  # 成功，返回可执行文件路径
    error = pyqtSignal(str)

    def __init__(self, target_bin_dir, prefer_gpu: bool = False):
        super().__init__()
        self.target_bin_dir = Path(target_bin_dir)
        self.prefer_gpu = prefer_gpu

    def run(self):
        try:
            self.progress.emit(0, "正在查询最新版本…")
            url = get_release_asset_url(self.prefer_gpu)
            logger.info("CrispASR 引擎下载链接: %s", url)

            self.target_bin_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp:
                zip_path = Path(tmp) / "crispasr.zip"
                self._download(url, zip_path)

                self.progress.emit(95, "正在解压…")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.target_bin_dir)

            exe = self._find_exe(self.target_bin_dir)
            if not exe:
                raise RuntimeError("解压后未找到 crispasr 可执行文件")

            self.progress.emit(100, "下载完成")
            logger.info("CrispASR 引擎已就绪: %s", exe)
            self.finished.emit(str(exe))
        except Exception as e:
            logger.exception("CrispASR 引擎下载失败")
            self.error.emit(str(e))

    def _download(self, url: str, dest: Path):
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url, headers={"User-Agent": "VideoCaptioner-CrispASR"}
        )
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 256
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 90)  # 留 10% 给解压
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        self.progress.emit(
                            pct, f"下载中 {mb:.1f}/{total_mb:.1f} MB"
                        )
                    else:
                        self.progress.emit(
                            50, f"下载中 {downloaded / 1024 / 1024:.1f} MB"
                        )

    @staticmethod
    def _find_exe(root: Path):
        return _find_exe(root)


def _find_exe(root: Path):
    exe_name = "crispasr.exe" if os.name == "nt" else "crispasr"
    direct = root / exe_name
    if direct.exists():
        return direct
    for p in root.rglob(exe_name):
        return p
    return None


def download_crisp_asr_engine_sync(target_bin_dir, prefer_gpu: bool = False, progress=None):
    """同步下载并解压 CrispASR 引擎（供转录线程内调用，不创建 QThread）。

    Args:
        target_bin_dir: 解压目标目录（resource/bin/CrispASR）
        prefer_gpu: 是否优先下载 GPU(CUDA) 版本
        progress: 可选回调 progress(pct:int, msg:str)

    Returns:
        str: crispasr 可执行文件路径
    """
    if progress is None:
        progress = lambda pct, msg: None

    target_bin_dir = Path(target_bin_dir)
    target_bin_dir.mkdir(parents=True, exist_ok=True)

    # 已存在则直接返回
    existing = _find_exe(target_bin_dir)
    if existing:
        return str(existing)

    progress(0, "正在查询 CrispASR 最新版本…")
    url = get_release_asset_url(prefer_gpu)
    logger.info("CrispASR 引擎下载链接: %s", url)

    ctx = ssl.create_default_context()
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "crispasr.zip"
        req = urllib.request.Request(
            url, headers={"User-Agent": "VideoCaptioner-CrispASR"}
        )
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 256
            with open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 90)
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        progress(pct, f"下载引擎 {mb:.1f}/{total_mb:.1f} MB")
                    else:
                        progress(50, f"下载引擎 {downloaded / 1024 / 1024:.1f} MB")

        progress(95, "正在解压引擎…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_bin_dir)

    exe = _find_exe(target_bin_dir)
    if not exe:
        raise RuntimeError("解压后未找到 crispasr 可执行文件")
    progress(100, "引擎下载完成")
    return str(exe)
