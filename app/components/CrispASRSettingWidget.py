from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QShowEvent
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import ComboBoxSettingCard
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    HyperlinkCard,
    InfoBar,
    InfoBarPosition,
    SettingCardGroup,
    SingleDirectionScrollArea,
    SwitchSettingCard,
)

from app.common.config import cfg
from app.config import MODEL_PATH
from app.core.entities import TranscribeLanguageEnum, WhisperModelEnum
from app.core.utils.logger import setup_logger

# CrispASR 与 WhisperCpp 共用 ggml 模型，复用其模型清单与下载对话框
from .WhisperCppSettingWidget import WHISPER_CPP_MODELS, WhisperCppDownloadDialog

logger = setup_logger("crisp_asr_setting")


def check_crisp_asr_models_exist() -> bool:
    """检查是否存在任意可用的 ggml 模型（CrispASR 与 WhisperCpp 共用）"""
    for model in WHISPER_CPP_MODELS:
        if (Path(MODEL_PATH) / model["value"]).exists():
            return True
    return False


class CrispASRSettingWidget(QWidget):
    """CrispASR 转录设置面板。

    CrispASR 是 whisper.cpp 的兼容分支，复用相同的 ggml-*.bin 模型，
    因此模型下载沿用 WhisperCpp 的下载对话框。UI 风格与其他转录后端保持一致。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._connect_signals()

    def showEvent(self, a0: QShowEvent) -> None:
        super().showEvent(a0)
        # 没有任何已下载模型时，提示用户先下载模型
        if not check_crisp_asr_models_exist():
            self.show_error_info(
                self.tr("未检测到可用模型，请先下载 ggml 模型（与 WhisperCpp 共用）")
            )
        return

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 创建单向滚动区域和容器
        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
        self.scrollArea.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )

        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        # ---------------- 模型设置组 ----------------
        self.setting_group = SettingCardGroup(
            self.tr("CrispASR 设置（✨本地推荐✨）"), self
        )

        # 模型选择（与 WhisperCpp 共用 ggml 模型）
        self.model_card = ComboBoxSettingCard(
            cfg.crisp_asr_model,
            FIF.ROBOT,
            self.tr("模型"),
            self.tr("选择模型（与 WhisperCpp 共用 ggml 模型）"),
            [model.value for model in WhisperModelEnum],
            self.setting_group,
        )

        # 仅显示已下载的模型
        for i in range(self.model_card.comboBox.count() - 1, -1, -1):
            model_text = self.model_card.comboBox.itemText(i).lower()
            model_config = next(
                (
                    model
                    for model in WHISPER_CPP_MODELS
                    if model["label"].lower() == model_text
                ),
                None,
            )
            if model_config and (MODEL_PATH / model_config["value"]).exists():
                continue
            self.model_card.comboBox.removeItem(i)

        # 模型管理（复用 WhisperCpp 模型下载对话框）
        self.manage_model_card = HyperlinkCard(
            "",  # 无链接
            self.tr("管理模型"),
            FIF.DOWNLOAD,  # 使用下载图标
            self.tr("模型管理"),
            self.tr("下载或更新 ggml 模型（CrispASR 与 WhisperCpp 共用）"),
            self.setting_group,
        )

        # 语言选择
        self.language_card = ComboBoxSettingCard(
            cfg.transcribe_language,
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("音频的源语言"),
            [language.value for language in TranscribeLanguageEnum],
            self.setting_group,
        )
        self.language_card.comboBox.setMaxVisibleItems(6)

        # ---------------- 其他设置组 ----------------
        self.other_group = SettingCardGroup(self.tr("其他设置"), self)

        # GPU 加速开关
        self.gpu_card = SwitchSettingCard(
            FIF.SPEED_HIGH,
            self.tr("GPU 加速"),
            self.tr("使用 GPU 加速转录（需要支持的显卡，默认关闭）"),
            cfg.crisp_asr_use_gpu,
            self.other_group,
        )

        # VAD 分段开关
        self.vad_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            self.tr("VAD 语音分段"),
            self.tr("使用 Silero VAD 进行语音分段，更适合字幕场景"),
            cfg.crisp_asr_use_vad,
            self.other_group,
        )

        # 添加模型设置组的卡片
        self.setting_group.addSettingCard(self.model_card)
        self.setting_group.addSettingCard(self.manage_model_card)
        self.setting_group.addSettingCard(self.language_card)

        # 添加其他设置组的卡片
        self.other_group.addSettingCard(self.gpu_card)
        self.other_group.addSettingCard(self.vad_card)

        # 将所有设置组添加到容器布局
        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addWidget(self.other_group)
        self.containerLayout.addStretch(1)

        # 设置组件最小宽度
        self.model_card.comboBox.setMinimumWidth(200)
        self.language_card.comboBox.setMinimumWidth(200)

        # 设置滚动区域
        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)

        # 将滚动区域添加到主布局
        self.main_layout.addWidget(self.scrollArea)

    def _connect_signals(self):
        """连接信号"""
        self.manage_model_card.linkButton.clicked.connect(self._show_model_manager)

    def _show_model_manager(self):
        """显示模型管理对话框（与 WhisperCpp 共用 ggml 模型）"""
        dialog = WhisperCppDownloadDialog(self.window(), self)
        dialog.show()

    def show_error_info(self, error_msg):
        """显示错误信息"""
        InfoBar.error(
            title=self.tr("提示"),
            content=error_msg,
            parent=self.window(),
            duration=5000,
            position=InfoBarPosition.BOTTOM,
        )
