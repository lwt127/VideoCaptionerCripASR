from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import ComboBoxSettingCard
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    HyperlinkCard,
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


class CrispASRSettingWidget(QWidget):
    """CrispASR 转录设置面板。

    CrispASR 是 whisper.cpp 的兼容分支，复用相同的 ggml-*.bin 模型，
    因此模型下载沿用 WhisperCpp 的下载对话框。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_signals()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
        self.scrollArea.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )

        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        self.setting_group = SettingCardGroup(
            self.tr("CrispASR 设置（本地, 复用 Whisper 模型）"), self
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
            model_configs = {
                model["label"].lower(): model for model in WHISPER_CPP_MODELS
            }
            model_config = model_configs.get(model_text)
            if model_config and (MODEL_PATH / model_config["value"]).exists():
                continue
            self.model_card.comboBox.removeItem(i)

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

        # GPU 加速开关
        self.gpu_card = SwitchSettingCard(
            FIF.SPEED_HIGH,
            self.tr("GPU 加速"),
            self.tr("使用 GPU 加速转录（需要支持的显卡, 默认关闭）"),
            cfg.crisp_asr_use_gpu,
            self.setting_group,
        )

        # VAD 分段开关
        self.vad_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            self.tr("VAD 语音分段"),
            self.tr("使用 Silero VAD 进行语音分段（更适合字幕场景）"),
            cfg.crisp_asr_use_vad,
            self.setting_group,
        )

        # 模型管理（复用 WhisperCpp 模型下载对话框）
        self.manage_model_card = HyperlinkCard(
            "",
            self.tr("管理模型"),
            FIF.DOWNLOAD,
            self.tr("模型管理"),
            self.tr("下载或更新 ggml 模型（CrispASR 与 WhisperCpp 共用）"),
            self.setting_group,
        )

        self.setting_group.addSettingCard(self.model_card)
        self.setting_group.addSettingCard(self.language_card)
        self.setting_group.addSettingCard(self.gpu_card)
        self.setting_group.addSettingCard(self.vad_card)
        self.setting_group.addSettingCard(self.manage_model_card)

        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addStretch(1)

        self.model_card.comboBox.setMinimumWidth(200)
        self.language_card.comboBox.setMinimumWidth(200)

        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)
        self.main_layout.addWidget(self.scrollArea)

    def setup_signals(self):
        self.manage_model_card.linkButton.clicked.connect(self.show_download_dialog)

    def show_download_dialog(self):
        """显示模型下载对话框（与 WhisperCpp 共用 ggml 模型）"""
        download_dialog = WhisperCppDownloadDialog(self.window(), self)
        download_dialog.show()
