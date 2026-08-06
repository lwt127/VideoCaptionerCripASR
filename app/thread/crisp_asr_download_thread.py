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


# 固定下载的 CrispASR 引擎版本（确定性升级；置为 None 时回退到 releases/latest）。
# v0.8.25 提供完整的 Windows CUDA 可执行包（crispasr-windows-x86_64-cuda.zip）。
CRISP_ASR_PINNED_TAG = "v0.8.25"


def _windows_asset_name(prefer_gpu: bool = False) -> str:
    """选择 Windows 平台的发行包资产名。默认使用自包含的 CPU 版（最稳）。"""
    if prefer_gpu:
        return "crispasr-windows-x86_64-cuda.zip"
    return "crispasr-windows-x86_64-cpu.zip"


def get_release_asset_url(prefer_gpu: bool = False) -> str:
    """查询 CrispASR Release，返回匹配当前平台的资产下载链接。

    默认锁定到 ``CRISP_ASR_PINNED_TAG``（确定性升级）；若该 tag 查询失败，
    则回退到 ``releases/latest``。
    """
    system = platform.system()
    if system != "Windows":
        raise RuntimeError(
            f"暂仅支持 Windows 自动下载 CrispASR 引擎，请手动构建（当前系统: {system}）"
        )

    asset_name = _windows_asset_name(prefer_gpu)

    ctx = ssl.create_default_context()

    def _fetch(api_url: str):
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "VideoCaptioner-CrispASR"}
        )
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # 优先使用固定 tag，失败（网络/该 tag 不存在）时回退 latest。
    data = None
    if CRISP_ASR_PINNED_TAG:
        try:
            data = _fetch(
                f"https://api.github.com/repos/{CRISP_ASR_REPO}/releases/tags/{CRISP_ASR_PINNED_TAG}"
            )
            logger.info("使用固定 CrispASR 版本: %s", CRISP_ASR_PINNED_TAG)
        except Exception as e:
            logger.warning(
                "查询固定版本 %s 失败，回退到 latest: %s", CRISP_ASR_PINNED_TAG, e
            )
    if data is None:
        data = _fetch(
            f"https://api.github.com/repos/{CRISP_ASR_REPO}/releases/latest"
        )

    assets = data.get("assets", [])

    def _find(pred):
        for a in assets:
            name = a.get("name", "")
            if name.startswith("crispasr-") and name.endswith(".zip") and pred(name):
                return a["browser_download_url"]
        return None

    # 1) 精确匹配
    url = _find(lambda n: n == asset_name)
    if url:
        return url

    # 2) GPU 优先：cuda → vulkan；否则 cpu
    if prefer_gpu:
        url = _find(lambda n: "windows" in n and "cuda" in n)
        if url:
            return url
        url = _find(lambda n: "windows" in n and "vulkan" in n)
        if url:
            return url

    # 3) 回退到 windows cpu（非 legacy 优先）
    url = _find(lambda n: "windows" in n and "cpu" in n and "legacy" not in n)
    if url:
        return url
    url = _find(lambda n: "windows" in n and "cpu" in n)
    if url:
        return url

    tag = CRISP_ASR_PINNED_TAG or "latest"
    raise RuntimeError(
        f"在 Release {tag} 中未找到 Windows 资产（期望: {asset_name}）"
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


def download_crisp_asr_engine_sync(
    target_bin_dir, prefer_gpu: bool = False, progress=None, force: bool = False
):
    """同步下载并解压 CrispASR 引擎（供转录线程内调用，不创建 QThread）。

    Args:
        target_bin_dir: 解压目标目录（resource/bin/CrispASR）
        prefer_gpu: 是否优先下载 GPU(CUDA) 版本
        progress: 可选回调 progress(pct:int, msg:str)
        force: 为 True 时即使已存在也重新下载（用于 CPU→CUDA 升级）

    Returns:
        str: crispasr 可执行文件路径
    """
    if progress is None:
        progress = lambda pct, msg: None

    target_bin_dir = Path(target_bin_dir)
    target_bin_dir.mkdir(parents=True, exist_ok=True)

    # 已存在且非强制时直接返回
    if not force:
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
        # 先解压到临时目录，再把含 crispasr.exe 的那一层“拍平”复制到目标目录，
        # 避免 GPU 包解压成子文件夹、而旧的 CPU exe 仍残留在顶层导致不生效。
        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        src_exe = _find_exe(extract_dir)
        if not src_exe:
            raise RuntimeError("解压后未找到 crispasr 可执行文件")
        src_dir = src_exe.parent

        import shutil as _shutil

        # 将 exe 所在目录的全部文件复制到目标目录（覆盖旧的 CPU 构建）
        for item in src_dir.iterdir():
            dest = target_bin_dir / item.name
            try:
                if item.is_dir():
                    _shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    _shutil.copy2(item, dest)
            except Exception as e:
                logger.warning("复制引擎文件失败 %s: %s", item.name, e)

    exe = target_bin_dir / ("crispasr.exe" if os.name == "nt" else "crispasr")
    if not exe.exists():
        exe = _find_exe(target_bin_dir)
    if not exe:
        raise RuntimeError("安装后未找到 crispasr 可执行文件")
    progress(100, "引擎下载完成")
    return str(exe)
