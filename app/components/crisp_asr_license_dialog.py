"""GUI-thread consent gate shared by single, batch and settings entry points."""

from html import escape

from PyQt5.QtCore import Qt
from qfluentwidgets import MessageBox, InfoBar

from app.common.config import cfg
from app.common.i18n import ui_text
from app.core.bk_asr.crisp_asr_catalog import resolve_model
from app.core.bk_asr.crisp_asr_licenses import acceptance_key, model_license
from app.core.entities import TranscribeConfig, TranscribeModelEnum


def ensure_crispasr_license(parent, config: TranscribeConfig) -> bool:
    if config.transcribe_model != TranscribeModelEnum.CRISP_ASR:
        return True
    try:
        backend, model = resolve_model(config.crisp_asr_backend or "", config.crisp_asr_model or "")
    except ValueError:
        InfoBar.error(
            ui_text("模型"), ui_text("模型选择已失效，请重新选择；不会自动改用其他模型"),
            parent=parent.window(), duration=5000,
        )
        return False
    license = model_license(backend, model)
    if license is None:
        return True

    saved = cfg.crisp_asr_license_acceptances.value
    saved = [key for key in saved if isinstance(key, str)] if isinstance(saved, list) else []
    key = acceptance_key(backend, model, license)
    if key not in saved:
        dialog = MessageBox(
            ui_text("确认模型授权"),
            escape(ui_text("模型：{backend} / {model}").format(backend=backend, model=model)) + "<br><br>"
            + escape(ui_text(license.notice)) + "<br><br>"
            + f'<a href="{escape(license.url, quote=True)}">{escape(ui_text("查看完整模型条款"))}</a><br><br>'
            + escape(ui_text("点击「我已阅读并同意」后才会运行该模型。取消不会开始任务。"
                             "确认仅适用于此模型及当前审核版本；不会接受其他模型的条款。")),
            parent.window(),
        )
        dialog.contentLabel.setTextFormat(Qt.RichText)
        dialog.contentLabel.setOpenExternalLinks(True)
        dialog.yesButton.setText(ui_text("我已阅读并同意"))
        dialog.cancelButton.setText(ui_text("取消"))
        if not dialog.exec():
            return False
        saved.append(key)
        cfg.set(cfg.crisp_asr_license_acceptances, saved)

    # Only attach the exact approval needed for this task snapshot.
    config.crisp_asr_license_acceptances = (key,)
    return True