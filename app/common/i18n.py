"""Supplemental UI translations for dynamic labels not covered by Qt catalogs.

This UTF-8 resource is loaded directly: changes do not require an external
Qt Linguist executable or modification of upstream compiled QM resources.
Never use translated labels as storage keys or protocol values.
"""

import json
import logging
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QLocale, QTranslator


def locale_key(locale: QLocale) -> str:
    if locale.language() == QLocale.Chinese:
        return "zh_TW" if locale.script() == QLocale.TraditionalHanScript else "zh_CN"
    return "en_US"


def ui_text(source: str) -> str:
    return QCoreApplication.translate("DynamicUI", source)


class SupplementalTranslator(QTranslator):
    def __init__(self, locale: QLocale, resource: Path, parent=None):
        super().__init__(parent)
        self.messages = {}
        try:
            data = json.loads(resource.read_text(encoding="utf-8"))
            self.messages = data.get(locale_key(locale), {})
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Unable to load supplemental UI translations", exc_info=True)

    def isEmpty(self):
        return not self.messages

    def translate(self, context, sourceText, disambiguation=None, n=-1):
        # Return None (null QString) on misses, allowing existing QM translators
        # to handle the source. Returning sourceText would shadow their entries.
        return self.messages.get(sourceText)