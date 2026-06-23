import os
import platform
import shutil
import ssl
import subprocess
import urllib.request

from PyQt5.QtCore import Qt, QThread, pyqtSignal

from app.config import CACHE_PATH
from app.core.utils.logger import setup_logger

logger = setup_logger("download_thread")


def _find_aria2c():
    """查找 aria2c 可执行文件（PATH 或随应用打包的 resource/bin）。未找到返回 None。"""
    found = shutil.which("aria2c")
    if found:
        return found
    try:
        from app.config import BIN_PATH

        name = "aria2c.exe" if os.name == "nt" else "aria2c"
        candidate = os.path.join(str(BIN_PATH), name)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return None


class FileDownloadThread(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, url, save_path):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.process = None
        self._stop_requested = False

    def run(self):
        try:
            # 创建缓存下载目录
            temp_dir = CACHE_PATH / "aria2c_download_cache"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / os.path.basename(self.save_path)

            # 检查是否存在未完成的下载文件
            if temp_file.exists():
                logger.info(f"发现未完成的下载文件: {temp_file}")
            self.progress.emit(0, self.tr("正在连接..."))

            # 优先使用 aria2c（多线程, 断点续传）；若不可用则回退到纯 Python 下载
            aria2c = _find_aria2c()
            if not aria2c:
                logger.info("未找到 aria2c，使用内置下载器")
                self._download_with_urllib(temp_file)
                return

            cmd = [
                aria2c,
                '--show-console-readout=false',
                '--summary-interval=1',
                '-x2',
                '-s2',
                '--connect-timeout=10',  # 连接超时时间10秒
                '--timeout=10',          # 数据传输超时时间10秒
                '--max-tries=2',         # 最大重试次数2次
                '--retry-wait=1',        # 重试等待时间1秒
                '--continue=true',       # 开启断点续传
                '--auto-file-renaming=false',
                '--allow-overwrite=true',
                '--check-certificate=false',                f'--dir={temp_dir}',
                f'--out={temp_file.name}',
                self.url
            ]
            
            # 根据操作系统设置不同的 subprocess 参数
            subprocess_args = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
                'universal_newlines': True,
                'encoding': 'utf-8'
            }
            
            # 仅在 Windows 系统上添加 CREATE_NO_WINDOW 标志
            if platform.system() == 'Windows':
                subprocess_args['creationflags'] = subprocess.CREATE_NO_WINDOW
            
            logger.info("运行下载命令: %s", " ".join(cmd))
            
            self.process = subprocess.Popen(
                cmd,
                **subprocess_args
            )
            
            while True:
                if self.process.poll() is not None:
                    break
                    
                line = self.process.stdout.readline()
                
                if '[#' in line and ']' in line:
                    try:
                        # 解析类似 "[#40ca1b 2.4MiB/74MiB(3%) CN:2 DL:3.9MiB ETA:18s]" 的格式
                        progress_part = line.split('(')[1].split(')')[0]
                        percent = float(progress_part.strip('%'))
                        
                        # 提取下载速度和剩余时间
                        speed = "0"
                        eta = ""
                        if "DL:" in line:
                            speed = line.split("DL:")[1].split()[0]
                        if "ETA:" in line:
                            eta = line.split("ETA:")[1].split(']')[0]
                        status_msg = f"{self.tr('速度')}: {speed}/s, {self.tr('剩余时间')}: {eta}"
                        self.progress.emit(percent, status_msg)
                    except Exception as e:
                        pass
                        
            if self.process.returncode == 0:
                # 下载完成后移动文件到目标位置
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                shutil.move(str(temp_file), self.save_path)
                self.finished.emit()
            else:
                error = self.process.stderr.read()
                logger.error("下载失败: %s", error)
                self.error.emit(f"{self.tr('下载失败')}: {error}")
                
        except Exception as e:
            logger.error("下载异常: %s", str(e))
            self.error.emit(str(e))

    def _download_with_urllib(self, temp_file):
        """纯 Python 下载回退（无需 aria2c），支持进度与中断。"""
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE  # 与 aria2c 的 --check-certificate=false 一致

            req = urllib.request.Request(
                self.url, headers={"User-Agent": "VideoCaptioner"}
            )
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                block = 1024 * 256
                with open(temp_file, "wb") as f:
                    while True:
                        if self._stop_requested:
                            logger.info("下载已被取消")
                            return
                        chunk = resp.read(block)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            percent = downloaded / total * 100
                            mb = downloaded / 1024 / 1024
                            total_mb = total / 1024 / 1024
                            self.progress.emit(
                                percent,
                                f"{mb:.1f}/{total_mb:.1f} MB",
                            )
                        else:
                            self.progress.emit(
                                0, f"{downloaded / 1024 / 1024:.1f} MB"
                            )

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            shutil.move(str(temp_file), self.save_path)
            self.finished.emit()
        except Exception as e:
            logger.error("内置下载器失败: %s", str(e))
            self.error.emit(f"{self.tr('下载失败')}: {e}")

    def stop(self):
        self._stop_requested = True
        if self.process:
            self.process.terminate()
            self.process.wait()
