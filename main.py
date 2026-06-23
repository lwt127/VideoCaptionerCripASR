"""VideoCaptioner application entry point.

Run with:  python main.py

Requires the dependencies in requirements.txt and (for transcription) the
CrispASR / Faster-Whisper backends described in the README.
"""

import os
import sys

from PyQt5.QtCore import Qt, QLocale, QTranslator
from PyQt5.QtWidgets import QApplication


def _install_translators(app: "QApplication") -> None:
    """根据用户在设置中选择的界面语言，安装相应的翻译器。

    源代码字符串为简体中文，因此：
      - 选择英文 → 加载 VideoCaptioner_en_US.qm
      - 选择繁体中文 → 加载 VideoCaptioner_zh_TW.qm
      - 选择简体中文 → 加载 VideoCaptioner_zh_CN.qm（与源相同，可选）
      - AUTO → 跟随系统区域
    """
    from app.common.config import Language, cfg
    from app.config import RESOURCE_PATH

    # 解析目标 locale
    lang = cfg.get(cfg.language)
    if lang == Language.AUTO:
        locale = QLocale.system()
    else:
        locale = lang.value  # QLocale 实例

    locale_name = locale.name()  # zh_CN / zh_TW / en_US ...

    # 1) qfluentwidgets 自带控件的翻译
    try:
        from qfluentwidgets import FluentTranslator

        fluent = FluentTranslator(locale)
        app.installTranslator(fluent)
        app._fluent_translator = fluent  # 持有引用，防止被回收
    except Exception:
        pass

    # 2) 应用自身的翻译（resource/translations/VideoCaptioner_<locale>.qm）
    qm_dir = os.path.join(str(RESOURCE_PATH), "translations")
    candidates = [f"VideoCaptioner_{locale_name}"]
    # 繁/简的回退（如系统返回 zh_HK 等）
    if locale_name.startswith("zh"):
        if "TW" in locale_name or "HK" in locale_name or "Hant" in locale_name:
            candidates.append("VideoCaptioner_zh_TW")
        else:
            candidates.append("VideoCaptioner_zh_CN")
    elif locale_name.startswith("en"):
        candidates.append("VideoCaptioner_en_US")

    translator = QTranslator(app)
    for base in candidates:
        if translator.load(base, qm_dir):
            app.installTranslator(translator)
            app._app_translator = translator  # 持有引用
            break


def main() -> int:
    # Enable High-DPI scaling for crisp UI on high-resolution displays.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # 在创建任何界面之前安装翻译器（按用户选择的界面语言）
    _install_translators(app)

    # Import after QApplication so Qt resources/config initialise correctly.
    from app.view.main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Make sure the project root is importable as the `app` package.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
