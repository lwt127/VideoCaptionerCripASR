"""Pinned, exact-flavor Windows runtime downloads with staged validation."""

import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.bk_asr.crisp_asr_catalog import CRISP_ASR_REPO
from app.core.utils.logger import setup_logger

logger = setup_logger("crisp_asr_download")
CRISP_ASR_PINNED_TAG = "v0.8.34"
WINDOWS_ASSETS = {
    flavor: f"crispasr-windows-x86_64-{flavor}.zip"
    for flavor in ("cpu", "cpu-legacy", "cuda", "cuda13", "vulkan")
}
_INSTALL_LOCK = threading.Lock()


def _windows_asset_name(prefer_gpu=False, flavor=None):
    flavor = flavor or ("cuda" if prefer_gpu else "cpu")
    if flavor not in WINDOWS_ASSETS:
        raise ValueError(f"未知引擎版本: {flavor}")
    return WINDOWS_ASSETS[flavor]


def get_release_asset_url(prefer_gpu=False, flavor=None):
    if platform.system() != "Windows":
        raise RuntimeError("暂仅支持 Windows 自动下载 CrispASR，请手动安装对应平台引擎")
    asset_name = _windows_asset_name(prefer_gpu, flavor)
    release = f"tags/{CRISP_ASR_PINNED_TAG}" if CRISP_ASR_PINNED_TAG else "latest"
    req = urllib.request.Request(
        f"https://api.github.com/repos/{CRISP_ASR_REPO}/releases/{release}",
        headers={"User-Agent": "VideoCaptioner-CrispASR"},
    )
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as response:
        data = json.load(response)
    for asset in data.get("assets", []):
        if asset.get("name") == asset_name:
            return asset["browser_download_url"]
    raise RuntimeError(f"Release {CRISP_ASR_PINNED_TAG} 缺少所选完整引擎包: {asset_name}")


def _find_exe(root):
    name = "crispasr.exe" if os.name == "nt" else "crispasr"
    direct = Path(root) / name
    return direct if direct.is_file() else next(Path(root).rglob(name), None)


def validate_engine(exe, flavor):
    result = subprocess.run(
        [str(exe), "--version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output = result.stdout + result.stderr
    match = re.search(r"version\s*:\s*(\d+\.\d+\.\d+)", output, re.I)
    if result.returncode or not match:
        raise RuntimeError("引擎无法启动；请检查 CPU 指令集及驱动，旧 CPU 可选 Legacy CPU。")
    if CRISP_ASR_PINNED_TAG and match.group(1) != CRISP_ASR_PINNED_TAG.lstrip("v"):
        raise RuntimeError("下载引擎版本与固定版本不一致")
    backend_match = re.search(r"ggml backends\s*:\s*([^\r\n]+)", output, re.I)
    backends = {x.strip() for x in backend_match.group(1).lower().split(",")} if backend_match else set()
    required = "cuda" if flavor in ("cuda", "cuda13") else "vulkan" if flavor == "vulkan" else "cpu"
    if required not in backends or (flavor.startswith("cpu") and backends != {"cpu"}):
        raise RuntimeError(f"引擎未包含所选后端: {flavor}")
    if flavor in ("cuda", "cuda13"):
        major = "13" if flavor == "cuda13" else "12"
        if not re.search(rf"cuda runtime ABI\s*:\s*{major}\b", output, re.I):
            raise RuntimeError(f"CUDA runtime ABI 与所选版本不一致: {flavor}")
    return output


def _install_archive(archive, target, flavor):
    """Validate a clean sibling directory, then swap with rollback on failure."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crispasr-stage-", dir=target.parent) as temp:
        root = Path(temp)
        with zipfile.ZipFile(archive) as zf:
            for entry in zf.infolist():
                if not (root / entry.filename).resolve().is_relative_to(root.resolve()):
                    raise RuntimeError("引擎压缩包包含不安全路径")
            zf.extractall(root)
        exe = _find_exe(root)
        if exe is None:
            raise RuntimeError("压缩包中没有 crispasr 可执行文件")
        if flavor in ("cuda", "cuda13"):
            major = "13" if flavor == "cuda13" else "12"
            required_files = ["ggml-cuda.dll", f"cudart64_{major}.dll",
                              f"cublas64_{major}.dll", f"cublasLt64_{major}.dll"]
            if any(not (exe.parent / name).is_file() for name in required_files):
                raise RuntimeError("CUDA 包缺少后端或运行库，保留原有安装")
        elif flavor == "vulkan" and not (exe.parent / "ggml-vulkan.dll").is_file():
            raise RuntimeError("Vulkan 包缺少后端 DLL，保留原有安装")
        validate_engine(exe, flavor)
        source = exe.parent
        if source == root:
            payload = root / "payload"
            payload.mkdir()
            for item in list(root.iterdir()):
                if item != payload:
                    shutil.move(str(item), payload / item.name)
            source = payload
        backup = target.with_name(target.name + ".backup-" + uuid.uuid4().hex)
        moved_old = False
        try:
            if target.exists():
                target.rename(backup)
                moved_old = True
            source.rename(target)
        except Exception:
            if moved_old and not target.exists():
                backup.rename(target)
            raise
        if moved_old:
            shutil.rmtree(backup, ignore_errors=True)
    return str(target / ("crispasr.exe" if os.name == "nt" else "crispasr"))


def download_crisp_asr_engine_sync(
    target_bin_dir, prefer_gpu=False, progress=None, force=False, flavor=None
):
    flavor = flavor or ("cuda" if prefer_gpu else "cpu")
    _windows_asset_name(flavor=flavor)
    progress = progress or (lambda pct, msg: None)
    target = Path(target_bin_dir)
    with _INSTALL_LOCK:
        existing = target / ("crispasr.exe" if os.name == "nt" else "crispasr")
        if not force and existing.is_file():
            try:
                validate_engine(existing, flavor)
                return str(existing)
            except (RuntimeError, OSError, subprocess.SubprocessError):
                pass
        progress(0, f"正在查询 CrispASR {flavor} 引擎…")
        url = get_release_asset_url(prefer_gpu, flavor=flavor)
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "engine.zip"
            request = urllib.request.Request(url, headers={"User-Agent": "VideoCaptioner-CrispASR"})
            with urllib.request.urlopen(request, timeout=60, context=ssl.create_default_context()) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with archive.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        downloaded += len(chunk)
                        progress(int(downloaded * 90 / total) if total else 50,
                                 f"下载引擎 {downloaded / 1024 / 1024:.1f} MB")
            progress(95, "正在验证及安装引擎…")
            exe = _install_archive(archive, target, flavor)
        progress(100, "引擎安装完成")
        return exe


class CrispASRDownloadThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, target_bin_dir, prefer_gpu=False, flavor=None):
        super().__init__()
        self.target_bin_dir = Path(target_bin_dir)
        self.prefer_gpu = prefer_gpu
        self.flavor = flavor

    def run(self):
        try:
            exe = download_crisp_asr_engine_sync(
                self.target_bin_dir, self.prefer_gpu, self.progress.emit, flavor=self.flavor
            )
            self.finished.emit(exe)
        except Exception as exc:
            logger.exception("CrispASR 引擎安装失败")
            self.error.emit(str(exc))

    _find_exe = staticmethod(_find_exe)