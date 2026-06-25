import logging
import threading
from pathlib import Path

from urllib3.exceptions import InsecureRequestWarning

from ...config import LOG_LEVEL, LOG_PATH


class _LevelSpecificFormatter(logging.Formatter):
    """INFO 级别使用简化格式，其余级别使用详细格式。

    线程安全：不再通过修改共享的 ``self._style._fmt`` 来切换格式（多线程下会相互
    覆盖），而是为两种格式分别持有独立的 Formatter 实例。
    """

    def __init__(self, info_fmt: str, default_fmt: str, datefmt: str):
        super().__init__(default_fmt, datefmt=datefmt)
        self._info_formatter = logging.Formatter(info_fmt, datefmt=datefmt)
        self._default_formatter = logging.Formatter(default_fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._info_formatter.format(record)
        return self._default_formatter.format(record)


# 全局唯一的处理器（进程内共享）。所有命名 logger 都通过 propagate 把记录交给根
# logger 上的这套处理器输出，避免每个命名 logger 各自持有一个指向同一个 app.log
# 的 RotatingFileHandler——多个文件句柄在 10MB 轮转(rollover)时相互争用，会导致
# 轮转失败、写入偏移错乱以及多字节 UTF-8 被截断（日志出现乱码、看似“停止记录”）。
_handlers_lock = threading.Lock()
_handlers_configured = False


def _configure_root_handlers(
    level: int,
    info_fmt: str,
    default_fmt: str,
    datefmt: str,
    log_file: str,
    console_output: bool,
) -> None:
    """只配置一次：在根 logger 上安装共享的控制台/文件处理器。"""
    global _handlers_configured
    with _handlers_lock:
        if _handlers_configured:
            return

        root = logging.getLogger()
        root.setLevel(level)

        level_formatter = _LevelSpecificFormatter(info_fmt, default_fmt, datefmt)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(level_formatter)
            root.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # 只使用单个 app.log（不产生 app.log.1 等轮转备份），且不限制大小。
            # mode="a"：追加写入，跨会话保留全部历史日志，文件大小不受限制；
            # 单一处理器实例 => 只有一个文件句柄，不存在并发轮转争用导致的编码损坏。
            file_handler = logging.FileHandler(
                log_file,
                mode="a",
                encoding="utf-8",
                delay=True,
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(level_formatter)
            root.addHandler(file_handler)

        _handlers_configured = True


def setup_logger(
    name: str,
    level: int = LOG_LEVEL,
    info_fmt: str = "%(message)s",  # INFO级别使用简化格式
    default_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 其他级别使用详细格式
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str = str(LOG_PATH / "app.log"),
    console_output: bool = True,
) -> logging.Logger:
    """
    创建并配置一个日志记录器，INFO级别使用简化格式。

    所有命名 logger 共享根 logger 上的同一套处理器（仅一个文件句柄），从而避免
    多个 RotatingFileHandler 指向同一文件时在轮转期间发生写入争用与编码损坏。

    参数：
    - name: 日志记录器的名称
    - level: 日志级别
    - info_fmt: INFO级别的日志格式字符串
    - default_fmt: 其他级别的日志格式字符串
    - datefmt: 时间格式字符串
    - log_file: 日志文件路径
    - console_output: 是否输出到控制台
    """
    _configure_root_handlers(
        level, info_fmt, default_fmt, datefmt, log_file, console_output
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 不在命名 logger 上挂处理器；让记录向上传播到根 logger 的共享处理器。
    logger.propagate = True

    # 兼容历史：如果此前某个命名 logger 自己挂过处理器（旧版本行为），清理掉，
    # 防止与共享处理器重复输出/重复持有文件句柄。
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = [
        "urllib3",
        "requests",
        "openai",
        "httpx",
        "httpcore",
        "ssl",
        "certifi",
    ]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logger
