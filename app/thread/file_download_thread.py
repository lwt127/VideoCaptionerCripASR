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
        """纯 Python 下载回退（无需 aria2c）。

        支持：HTTP Range 断点续传、进度/速度显示、可中断。
        若服务器不支持 Range，则重新从头下载。
        """
        import time

        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE  # 与 aria2c 的 --check-certificate=false 一致

            # 断点续传：已存在部分文件则从断点继续
            resume_from = 0
            if os.path.exists(temp_file):
                resume_from = os.path.getsize(temp_file)
                if resume_from > 0:
                    logger.info("尝试断点续传，已下载 %d 字节", resume_from)

            headers = {"User-Agent": "Mozilla/5.0 (VideoCaptioner)"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(self.url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=30, context=ctx)

            # 服务器是否接受断点续传（206=部分内容）
            if resume_from > 0 and resp.status != 206:
                logger.info("服务器不支持断点续传，重新下载")
                resume_from = 0

            content_len = int(resp.headers.get("Content-Length", 0))
            total = content_len + resume_from if content_len else 0

            mode = "ab" if resume_from > 0 else "wb"
            downloaded = resume_from
            block = 1024 * 512
            last_t = time.time()
            last_bytes = downloaded

            with resp, open(temp_file, mode) as f:
                while True:
                    if self._stop_requested:
                        logger.info("下载已被取消（已保留断点, 可续传）")
                        return
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_t >= 0.5:
                        speed = (downloaded - last_bytes) / (now - last_t)
                        last_t, last_bytes = now, downloaded
                        speed_str = self._fmt_speed(speed)
                        if total > 0:
                            percent = downloaded / total * 100
                            self.progress.emit(
                                percent,
                                f"{downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB  "
                                f"{self.tr('速度')}: {speed_str}",
                            )
                        else:
                            self.progress.emit(
                                0,
                                f"{downloaded/1024/1024:.1f} MB  "
                                f"{self.tr('速度')}: {speed_str}",
                            )

            self.progress.emit(100, self.tr("下载完成"))
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            shutil.move(str(temp_file), self.save_path)
            self.finished.emit()
        except Exception as e:
            logger.error("内置下载器失败: %s", str(e))
            self.error.emit(f"{self.tr('下载失败')}: {e}")

    @staticmethod
    def _fmt_speed(bytes_per_sec: float) -> str:
        if bytes_per_sec >= 1024 * 1024:
            return f"{bytes_per_sec/1024/1024:.1f} MB/s"
        if bytes_per_sec >= 1024:
            return f"{bytes_per_sec/1024:.0f} KB/s"
        return f"{bytes_per_sec:.0f} B/s"

    def stop(self):
        self._stop_requested = True
        if self.process:
            self.process.terminate()
            self.process.wait()
