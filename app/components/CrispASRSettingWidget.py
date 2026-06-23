from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import ComboBox, ComboBoxSettingCard
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    HyperlinkCard,
    SettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
    SwitchSettingCard,
)

from app.common.config import cfg
from app.config import BIN_PATH
from app.core.bk_asr.crisp_asr_catalog import (
    CRISP_ASR_PROJECT_URL,
    get_backend_labels,
    get_model_labels,
    get_vad_labels,
)
from app.core.entities import TranscribeLanguageEnum
from app.core.utils.logger import setup_logger

logger = setup_logger("crisp_asr_setting")

CRISP_ASR_BIN = Path(BIN_PATH) / "CrispASR" / "crispasr.exe"


class ComboSettingCard(SettingCard):
    """带下拉框的设置卡片（绑定到普通字符串 ConfigItem，支持动态选项）。

    与 qfluentwidgets 的 ComboBoxSettingCard 外观一致，但不要求 OptionsConfigItem，
    因此可用于选项随其他设置联动变化的场景（如随后端联动的模型列表）。
    """

    currentTextChanged = pyqtSignal(str)

    def __init__(self, config_item, icon, title, content=None, items=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.config_item = config_item
        self.comboBox = ComboBox(self)
        self.hBoxLayout.addWidget(self.comboBox, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

        if items:
            self.comboBox.addItems(items)

        # 初始化当前值
        saved = cfg.get(config_item)
        if saved and self.comboBox.findText(saved) >= 0:
            self.comboBox.setCurrentText(saved)

        self.comboBox.currentTextChanged.connect(self._on_changed)

    def _on_changed(self, text: str):
        if text:
            cfg.set(self.config_item, text)
            self.currentTextChanged.emit(text)

    def set_items(self, items, keep_saved=True):
        """重设下拉项；keep_saved 时尽量保持已保存值，否则选第一项并写回配置"""
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.comboBox.addItems(items)
        self.comboBox.blockSignals(False)
        if not items:
            return
        saved = cfg.get(self.config_item)
        if keep_saved and saved in items:
            self.comboBox.setCurrentText(saved)
        else:
            self.comboBox.setCurrentText(items[0])
            cfg.set(self.config_item, items[0])


class CrispASRSettingWidget(QWidget):
    """CrispASR 转录设置面板（多后端 ASR 引擎）。

    UI 与其他转录后端保持一致：后端引擎下拉、随后端联动的模型下拉、源语言、
    VAD 切片开关、VAD 方法下拉，以及引擎状态卡片。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._init_values()
        self._connect_signals()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # 单向滚动区域和容器
        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)
        self.scrollArea.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )
        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        # ---------------- CrispASR 设置组 ----------------
        self.setting_group = SettingCardGroup(self.tr("CrispASR 设置"), self)

        # 后端引擎选择（识别架构）
        self.backend_card = ComboSettingCard(
            cfg.crisp_asr_backend,
            FIF.ROBOT,
            self.tr("后端引擎"),
            self.tr("选择 CrispASR 后端（识别架构）"),
            get_backend_labels(),
            self.setting_group,
        )

        # 模型选择（随后端联动）
        self.model_card = ComboSettingCard(
            cfg.crisp_asr_model,
            FIF.DOWNLOAD,
            self.tr("模型"),
            self.tr("该后端可用的模型，标“自动下载”的会在首次运行时下载"),
            [],  # 初始为空，根据后端动态填充
            self.setting_group,
        )

        # 源语言
        self.language_card = ComboBoxSettingCard(
            cfg.transcribe_language,
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("音视频中说话的语言，默认自动识别"),
            [language.value for language in TranscribeLanguageEnum],
            self.setting_group,
        )
        self.language_card.comboBox.setMaxVisibleItems(6)

        # VAD 切片开关
        self.vad_card = SwitchSettingCard(
            FIF.CHECKBOX,
            self.tr("VAD 切片"),
            self.tr("使用语音活动检测切分长音频，长视频强烈建议开启"),
            cfg.crisp_asr_use_vad,
            self.setting_group,
        )

        # VAD 方法
        self.vad_method_card = ComboSettingCard(
            cfg.crisp_asr_vad_method,
            FIF.VOLUME,
            self.tr("VAD 方法"),
            self.tr("选择语音活动检测模型，默认 Silero"),
            get_vad_labels(),
            self.setting_group,
        )

        # 引擎状态 / 项目主页
        engine_ok = CRISP_ASR_BIN.exists()
        engine_desc = (
            self.tr("已检测到 crispasr 引擎，模型将在首次使用时自动下载")
            if engine_ok
            else self.tr("引擎与模型将在开始转录时自动下载，无需手动安装")
        )
        self.engine_card = HyperlinkCard(
            CRISP_ASR_PROJECT_URL,
            self.tr("项目主页"),
            FIF.GLOBE,
            self.tr("CrispASR 引擎"),
            engine_desc,
            self.setting_group,
        )

        # 添加卡片
        self.setting_group.addSettingCard(self.backend_card)
        self.setting_group.addSettingCard(self.model_card)
        self.setting_group.addSettingCard(self.language_card)
        self.setting_group.addSettingCard(self.vad_card)
        self.setting_group.addSettingCard(self.vad_method_card)
        self.setting_group.addSettingCard(self.engine_card)

        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addStretch(1)

        # 最小宽度
        self.backend_card.comboBox.setMinimumWidth(260)
        self.model_card.comboBox.setMinimumWidth(260)
        self.language_card.comboBox.setMinimumWidth(200)
        self.vad_method_card.comboBox.setMinimumWidth(260)

        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)
        self.main_layout.addWidget(self.scrollArea)

    def _init_values(self):
        """初始化下拉框默认值与联动状态"""
        backend_labels = get_backend_labels()
        saved_backend = cfg.crisp_asr_backend.value
        if saved_backend not in backend_labels:
            saved_backend = backend_labels[0]
            cfg.set(cfg.crisp_asr_backend, saved_backend)
        self.backend_card.comboBox.setCurrentText(saved_backend)

        # 根据后端填充模型列表（尽量保持已保存模型）
        self.model_card.set_items(get_model_labels(saved_backend), keep_saved=True)

        # VAD 方法默认值
        vad_labels = get_vad_labels()
        saved_vad = cfg.crisp_asr_vad_method.value
        if saved_vad not in vad_labels:
            saved_vad = vad_labels[0]
            cfg.set(cfg.crisp_asr_vad_method, saved_vad)
        self.vad_method_card.comboBox.setCurrentText(saved_vad)

        # VAD 方法可用性随 VAD 开关联动
        self._on_vad_toggled(cfg.crisp_asr_use_vad.value)

    def _connect_signals(self):
        """连接信号"""
        self.backend_card.currentTextChanged.connect(self._on_backend_changed)
        self.vad_card.checkedChanged.connect(self._on_vad_toggled)

    def _on_backend_changed(self, backend_label: str):
        """后端切换：刷新模型列表"""
        self.model_card.set_items(get_model_labels(backend_label), keep_saved=False)

    def _on_vad_toggled(self, checked: bool):
        """VAD 开关联动 VAD 方法卡片可用性"""
        self.vad_method_card.setEnabled(checked)
