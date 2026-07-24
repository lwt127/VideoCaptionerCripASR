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
    get_language_codes,
    get_lid_labels,
    get_model_labels,
    get_vad_labels,
    needs_lid_pre_step,
    supports_auto_lid,
)
from app.core.entities import LANGUAGES, TranscribeLanguageEnum
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


# 语言标签（中/英文皆有）→ ISO 639-1 代码的反查表，用于将 catalog 里的语言代码
# 子集映射回 TranscribeLanguageEnum 成员。
_LABEL_TO_CODE = LANGUAGES


def _enum_members_for_codes(codes, include_auto: bool = False):
    """按 TranscribeLanguageEnum 原始顺序，筛出代码在 codes 中的成员。

    codes 为 None 表示不限制，返回完整枚举列表（含“自动检测”，因为不限制通常
    意味着该后端/模型支持多语言/自动识别）。codes 为具体列表时按 include_auto
    决定是否在结果最前面加入“自动检测”选项（仅当后端/模型具备原生 LID 能力时
    才应传 True，见 supports_auto_lid）。
    """
    if codes is None:
        return list(TranscribeLanguageEnum)
    code_set = set(codes)
    result = [
        member
        for member in TranscribeLanguageEnum
        if member is not TranscribeLanguageEnum.AUTO
        and _LABEL_TO_CODE.get(member.value) in code_set
    ]
    if not result:
        return list(TranscribeLanguageEnum)
    if include_auto:
        result.insert(0, TranscribeLanguageEnum.AUTO)
    return result


# “自动检测”在需要额外 LID 预处理步骤的后端（cohere/canary/granite/voxtral/
# voxtral4b）上显示的专用文案，用于和原生自动识别的“自动检测”区分开。
_AUTO_LID_PRE_STEP_LABEL = "自动检测 (LID 预处理)"


class LanguageComboSettingCard(ComboBoxSettingCard):
    """源语言设置卡：支持根据 ASR 后端/模型动态收窄可选语言子集。

    qfluentwidgets 的 ComboBoxSettingCard.optionToText 在构造时基于完整的
    configItem.options 一次性建立，没有官方方式隐藏子集且保留
    OptionsConfigItem/EnumSerializer 绑定；这里通过重建 comboBox 选项 +
    optionToText 字典来实现动态收窄，同时仍写回同一个 OptionsConfigItem。
    """

    def set_items(self, members, keep_saved=True, auto_label: "str | None" = None):
        """重设可选语言枚举成员列表。

        keep_saved 为 True 且已保存值在新子集中时保留原值，否则回退到子集
        第一项并写回配置（与 ComboSettingCard.set_items 行为一致）。auto_label 不为
        None 时覆盖 AUTO 成员在下拉中展示的文本（但仍写入同一 configItem 值），
        用于区分“原生自动识别”与“LID 预处理”两种自动检测场景。
        """
        if not members:
            members = list(TranscribeLanguageEnum)

        texts = [self.tr(member.value) for member in members]
        if auto_label and TranscribeLanguageEnum.AUTO in members:
            idx = members.index(TranscribeLanguageEnum.AUTO)
            texts[idx] = auto_label
        self.optionToText = {o: t for o, t in zip(members, texts)}

        saved = cfg.get(self.configItem)
        if not (keep_saved and saved in members):
            saved = members[0]
            cfg.set(self.configItem, saved)

        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        for text, option in zip(texts, members):
            self.comboBox.addItem(text, userData=option)
        self.comboBox.setCurrentText(self.optionToText[saved])
        self.comboBox.blockSignals(False)


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

        # 源语言（随后端/模型联动，仅展示该组合实际支持的语言子集）
        self.language_card = LanguageComboSettingCard(
            cfg.transcribe_language,
            FIF.LANGUAGE,
            self.tr("源语言"),
            self.tr("音视频中说话的语言，默认自动识别；可选项随所选后端/模型自动收窄"),
            [language.value for language in TranscribeLanguageEnum],
            self.setting_group,
        )
        self.language_card.comboBox.setMaxVisibleItems(6)

        # LID 预处理方法（仅当后端无原生 LID 能力且选择了“自动检测”时可用）
        self.lid_method_card = ComboSettingCard(
            cfg.crisp_asr_lid_method,
            FIF.ROBOT,
            self.tr("LID 预处理模型"),
            self.tr(
                "该后端无原生语种识别能力，需额外模型先检测语种再转录，默认 FireRed"
            ),
            get_lid_labels(),
            self.setting_group,
        )

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
            self.tr(
                "选择语音活动检测模型，默认 Silero；"
                "日语 ASMR/轻声等安静音频可选 Whisper-VAD ASMR"
            ),
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
        self.setting_group.addSettingCard(self.lid_method_card)
        self.setting_group.addSettingCard(self.vad_card)
        self.setting_group.addSettingCard(self.vad_method_card)
        self.setting_group.addSettingCard(self.engine_card)

        self.containerLayout.addWidget(self.setting_group)
        self.containerLayout.addStretch(1)

        # 最小宽度
        self.backend_card.comboBox.setMinimumWidth(260)
        self.model_card.comboBox.setMinimumWidth(260)
        self.language_card.comboBox.setMinimumWidth(200)
        self.lid_method_card.comboBox.setMinimumWidth(220)
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

        # 根据后端+模型收窄语言下拉（尽量保持已保存语言）
        saved_model = self.model_card.comboBox.currentText()
        self._refresh_language_and_lid(saved_backend, saved_model, keep_saved=True)

        # LID 方法默认值
        lid_labels = get_lid_labels()
        saved_lid = cfg.crisp_asr_lid_method.value
        if saved_lid not in lid_labels:
            saved_lid = lid_labels[0]
            cfg.set(cfg.crisp_asr_lid_method, saved_lid)
        self.lid_method_card.comboBox.setCurrentText(saved_lid)

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
        self.model_card.currentTextChanged.connect(self._on_model_changed)
        # 注意：连接 currentIndexChanged 而非 currentTextChanged。qfluentwidgets 的
        # ComboBoxBase.setCurrentIndex 先 emit currentTextChanged 再 emit
        # currentIndexChanged，而 ComboBoxSettingCard._onCurrentIndexChanged（写入
        # qconfig 的那个方法）恰位于后者。若这里连接 currentTextChanged，
        # cfg.transcribe_language.value 在回调时还是上一个值，导致 LID 卡片
        # 启用状态慢一拍。
        self.language_card.comboBox.currentIndexChanged.connect(self._on_language_changed)
        self.vad_card.checkedChanged.connect(self._on_vad_toggled)

    def _on_backend_changed(self, backend_label: str):
        """后端切换：刷新模型列表 + 语言列表（模型列表刷新会触发 _on_model_changed，
        但为避免时序问题这里也直接刷新一次语言列表，使用新后端的第一个模型）"""
        self.model_card.set_items(get_model_labels(backend_label), keep_saved=False)
        model_label = self.model_card.comboBox.currentText()
        self._refresh_language_and_lid(backend_label, model_label, keep_saved=False)

    def _on_model_changed(self, model_label: str):
        """模型切换：刷新语言列表（同一后端下不同模型支持的语言可能不同）"""
        backend_label = self.backend_card.comboBox.currentText()
        self._refresh_language_and_lid(backend_label, model_label, keep_saved=True)

    def _on_language_changed(self, _index: int):
        """源语言切换：刷新 LID 预处理卡片可用性（只有选中“自动检测”且
        当前后端/模型需要 LID 预处理时才启用）。

        接收 currentIndexChanged 的 index 参数仅用于触发回调，实际判断直接读
        comboBox.itemData(index) 获取刚刚选中的枚举成员，而不依赖
        cfg.transcribe_language.value（它会在 ComboBoxSettingCard._onCurrentIndexChanged
        写入后才更新，与本回调的触发顺序存在竞态风险）。"""
        backend_label = self.backend_card.comboBox.currentText()
        model_label = self.model_card.comboBox.currentText()
        selected = self.language_card.comboBox.itemData(_index)
        self._update_lid_card_enabled(backend_label, model_label, selected_language=selected)

    def _language_members_for(self, backend_label: str, model_label: str):
        """根据后端标签 + 模型标签，解析出应展示的 TranscribeLanguageEnum 子集。

        若该后端/模型具备原生语种自动识别（native audio-LID）能力，或需要额外 LID
        预处理步骤才能实现自动检测（cohere/canary/granite/voxtral/voxtral4b），
        在列表最前面加入“自动检测”选项（对应 CrispASR 的 -l auto）。
        """
        codes = get_language_codes(backend_label, model_label or None)
        include_auto = supports_auto_lid(backend_label, model_label or None) or needs_lid_pre_step(
            backend_label, model_label or None
        )
        return _enum_members_for_codes(codes, include_auto=include_auto)

    def _refresh_language_and_lid(self, backend_label: str, model_label: str, keep_saved: bool):
        """刷新语言下拉（若需 LID 预处理则自动检测选项显示为“自动检测 (LID 预处理)”），
        并同步刷新 LID 卡片可用性。"""
        members = self._language_members_for(backend_label, model_label)
        auto_label = (
            self.tr(_AUTO_LID_PRE_STEP_LABEL)
            if needs_lid_pre_step(backend_label, model_label or None)
            else None
        )
        self.language_card.set_items(members, keep_saved=keep_saved, auto_label=auto_label)
        self._update_lid_card_enabled(backend_label, model_label)

    def _update_lid_card_enabled(
        self, backend_label: str, model_label: str, selected_language=None
    ):
        """LID 预处理卡片仅在“该后端/模型需要 LID 预处理”且“当前选中自动检测”时可用，
        否则置灰（不隐藏，保持布局稳定）。

        selected_language 不为 None 时优先使用（由调用方直接从 comboBox.itemData
        读取，避免与 cfg.transcribe_language.value 的写入时序竞争）；否则回退读取
        cfg.transcribe_language.value（适用于调用方已确保配置已同步写入的场景，
        如 set_items 内部已先 cfg.set 再返回的路径）。
        """
        needs_pre_step = needs_lid_pre_step(backend_label, model_label or None)
        current = (
            selected_language if selected_language is not None else cfg.transcribe_language.value
        )
        is_auto = current is TranscribeLanguageEnum.AUTO
        self.lid_method_card.setEnabled(needs_pre_step and is_auto)

    def _on_vad_toggled(self, checked: bool):
        """VAD 开关联动 VAD 方法卡片可用性"""
        self.vad_method_card.setEnabled(checked)
