import os
import re
import time
from pathlib import Path

import requests
import yt_dlp
from PyQt5.QtCore import QThread, pyqtSignal

from app.config import APPDATA_PATH
from app.core.entities import VideoInfo
from app.core.utils.logger import setup_logger

logger = setup_logger("video_download_thread")


class VideoDownloadThread(QThread):
    """视频下载线程类"""

    finished = pyqtSignal(
        str
    )  # 发送下载完成的信号(视频路径, 字幕路径, 缩略图路径, 视频信息)
    progress = pyqtSignal(int, str)  # 发送下载进度的信号
    error = pyqtSignal(str)  # 发送错误信息的信号

    def __init__(self, url: str, work_dir: str):
        super().__init__()
        self.url = url
        self.work_dir = work_dir

    def run(self):
        try:
            video_file_path, subtitle_file_path, thumbnail_file_path, info_dict = (
                self.download()
            )
            self.finished.emit(video_file_path)
        except Exception as e:
            error_message = self._format_download_error(e)
            logger.exception("下载视频失败: %s", error_message)
            self.error.emit(error_message)

    def _format_download_error(self, error: Exception) -> str:
        """Turn common YouTube authentication failures into an actionable message."""
        error_text = str(error)
        if (
            "Sign in to confirm" in error_text
            or "Use --cookies-from-browser" in error_text
            or "The page needs to be reloaded" in error_text
        ):
            cookiefile_path = APPDATA_PATH / "cookies.txt"
            return (
                "YouTube 登录验证已过期或未通过反机器人检查。请在浏览器中重新打开并刷新视频页面，"
                f"然后重新导出 Netscape 格式 cookies，覆盖保存为 "
                f"{cookiefile_path}，然后重试。"
                "导出和下载必须使用同一个网络连接；不要直接复制浏览器数据库文件。"
            )
        return error_text

    def progress_hook(self, d):
        """下载进度回调函数"""
        if d["status"] == "downloading":
            percent = d["_percent_str"]
            speed = d["_speed_str"]

            # 提取百分比和速度的纯文本
            clean_percent = (
                percent.replace("\x1b[0;94m", "")
                .replace("\x1b[0m", "")
                .strip()
                .replace("%", "")
            )
            clean_speed = speed.replace("\x1b[0;32m", "").replace("\x1b[0m", "").strip()

            self.progress.emit(
                int(float(clean_percent)),
                f"下载进度: {clean_percent}%  速度: {clean_speed}",
            )

    def sanitize_filename(self, name: str, replacement: str = "_") -> str:
        """清理文件名中不允许的字符"""
        # 定义不允许的字符
        forbidden_chars = r'<>:"/\\|?*'

        # 替换不允许的字符
        sanitized = re.sub(f"[{re.escape(forbidden_chars)}]", replacement, name)

        # 移除控制字符
        sanitized = re.sub(r"[\0-\31]", "", sanitized)

        # 去除文件名末尾的空格和点
        sanitized = sanitized.rstrip(" .")

        # 限制文件名长度
        max_length = 255
        if len(sanitized) > max_length:
            base, ext = os.path.splitext(sanitized)
            base_max_length = max_length - len(ext)
            sanitized = base[:base_max_length] + ext

        # 处理Windows保留名称
        windows_reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        name_without_ext = os.path.splitext(sanitized)[0].upper()
        if name_without_ext in windows_reserved_names:
            sanitized = f"{sanitized}_"

        # 如果文件名为空，返回默认名称
        if not sanitized:
            sanitized = "default_filename"

        return sanitized

    def download_automatic_subtitle(
        self, subtitle_url: str, language: str, subtitle_dir: Path
    ) -> str | None:
        """Download an automatic VTT subtitle without failing the video download.

        Subtitle endpoints can temporarily respond with HTTP 429 independently of
        the video stream. Retry those responses with backoff, then keep the video
        result when subtitles remain unavailable.
        """
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        subtitle_path = subtitle_dir / f"【下载字幕】{language}.vtt"
        headers = {"User-Agent": "Mozilla/5.0"}

        for attempt, delay in enumerate((5, 15, 30), start=1):
            try:
                response = requests.get(subtitle_url, headers=headers, timeout=30)
                response.raise_for_status()
                subtitle_path.write_text(response.text, encoding="utf-8")
                return str(subtitle_path)
            except requests.HTTPError as error:
                status_code = error.response.status_code if error.response else None
                if status_code != 429 or attempt == 3:
                    logger.warning("无法下载自动字幕: %s", error)
                    return None
                logger.warning(
                    "自动字幕请求被限流（第 %s/3 次），%s 秒后重试", attempt, delay
                )
            except requests.RequestException as error:
                logger.warning("无法下载自动字幕: %s", error)
                return None

            time.sleep(delay)

        return None

    def download_video(self, ydl_opts: dict) -> dict:
        """Download the video, refreshing expiring media URLs after HTTP 403."""
        for attempt in range(2):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(self.url, download=True)
            except yt_dlp.utils.DownloadError as error:
                is_forbidden = "HTTP Error 403" in str(error)
                if not is_forbidden or attempt:
                    raise
                logger.warning("视频下载链接已失效，正在刷新链接后重试")

        raise RuntimeError("视频下载重试失败")

    def download(self, need_subtitle: bool = True, need_thumbnail: bool = False):
        """下载视频"""
        logger.info("开始下载视频: %s", self.url)

        # 初始化 ydl 选项
        initial_ydl_opts = {
            "outtmpl": {
                "default": "%(title)s.%(ext)s",
                "subtitle": "【下载字幕】.%(ext)s",
                "thumbnail": "thumbnail",
            },
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",  # 优先mp4，回退到任意最佳视频+音频
            "merge_output_format": "mp4",  # 合并后输出mp4
            "progress_hooks": [self.progress_hook],  # 下载进度钩子
            "quiet": True,  # 禁用日志输出
            "no_warnings": True,  # 禁用警告信息
            "noprogress": True,
            # Retry transient transfer failures. A second extract_info call below
            # refreshes YouTube's short-lived media URL after an HTTP 403.
            "retries": 3,
            "fragment_retries": 3,
            "extractor_retries": 3,
            "file_access_retries": 3,
            "sleep_interval_requests": 1,
            "max_sleep_interval_requests": 5,
            # Download subtitles separately after the video. A subtitle-only
            # HTTP 429 must not discard an otherwise successful video download.
            "writeautomaticsub": False,
            "writethumbnail": need_thumbnail,  # 下载缩略图
            "thumbnail_format": "jpg",  # 指定缩略图的格式
        }

        # 检查 cookies 文件
        cookiefile_path = APPDATA_PATH / "cookies.txt"
        if cookiefile_path.exists():
            logger.info(f"使用cookiefile: {cookiefile_path}")
            initial_ydl_opts["cookiefile"] = str(cookiefile_path)

        with yt_dlp.YoutubeDL(initial_ydl_opts) as ydl:
            # 提取视频信息（不下载）
            info_dict = ydl.extract_info(self.url, download=False)

            # 设置动态下载文件夹为视频标题
            video_title = self.sanitize_filename(info_dict.get("title", "MyVideo"))
            video_work_dir = Path(self.work_dir) / self.sanitize_filename(video_title)
            subtitle_language = info_dict.get("language", None)
            if subtitle_language:
                subtitle_language = subtitle_language.lower().split("-")[0]

            subtitle_download_link = None
            selected_subtitle_language = None
            if need_subtitle and subtitle_language:
                for language, formats in info_dict.get("automatic_captions", {}).items():
                    if language.startswith(subtitle_language) and formats:
                        subtitle_download_link = formats[-1].get("url")
                        selected_subtitle_language = language
                        break

            # Extract again during the actual download so yt-dlp gets a fresh
            # signed media URL instead of reusing the earlier metadata response.
            download_ydl_opts = {
                **initial_ydl_opts,
                "paths": {
                    "home": str(video_work_dir),
                    "subtitle": str(video_work_dir / "subtitle"),
                    "thumbnail": str(video_work_dir),
                },
            }
            info_dict = self.download_video(download_ydl_opts)

            # 获取视频文件路径
            with yt_dlp.YoutubeDL(download_ydl_opts) as download_ydl:
                video_file_path = Path(download_ydl.prepare_filename(info_dict))
            if video_file_path.exists():
                video_file_path = str(video_file_path)
            else:
                video_file_path = None

            # A subtitle rate limit is non-fatal: return the downloaded video
            # and leave subtitle_file_path as None when the retries are exhausted.
            subtitle_file_path = None
            if subtitle_download_link and selected_subtitle_language:
                subtitle_file_path = self.download_automatic_subtitle(
                    subtitle_download_link,
                    selected_subtitle_language,
                    video_work_dir / "subtitle",
                )

            # 获取缩略图文件路径
            thumbnail_file_path = None
            for file in video_work_dir.glob("**/thumbnail*"):
                thumbnail_file_path = str(file)
                break

            logger.info(f"视频下载完成: {video_file_path}")
            logger.info(f"字幕文件路径: {subtitle_file_path}")
            return video_file_path, subtitle_file_path, thumbnail_file_path, info_dict
