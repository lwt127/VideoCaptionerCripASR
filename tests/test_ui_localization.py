"""UI locale and stable-selection regressions; never save real user settings."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QLocale, QTranslator
from PyQt5.QtWidgets import QApplication

from app.common.config import Language, LanguageSerializer, cfg
from app.common.i18n import SupplementalTranslator, locale_key, ui_text
from app.core.bk_asr.crisp_asr_catalog import (
    CRISP_ASR_BACKENDS, CRISP_ASR_LID_METHODS, CRISP_ASR_VAD_METHODS,
    get_language_codes, model_id, normalize_selection, resolve_model,
    resolve_lid_method, resolve_vad_method,
)
from app.core.bk_asr.crisp_asr_options import ENGINE_FLAVORS, ALIGNMENT_MODES, ALIGNERS
from app.core.bk_asr.crisp_asr_licenses import MODEL_LICENSES
from app.core.entities import TranscribeConfig, TranscribeModelEnum

RESOURCE = Path(__file__).resolve().parents[1] / "resource/translations/supplemental_ui.json"


class LocaleTests(unittest.TestCase):
    def test_auto_is_distinct_on_supported_system_locales(self):
        old = QLocale()
        try:
            for name in ("en_US", "zh_TW", "zh_CN", "zh_HK"):
                QLocale.setDefault(QLocale(name))
                self.assertEqual(len(Language), 4)
                self.assertEqual(Language.AUTO.value, "Auto")
                self.assertNotEqual(Language.AUTO, Language.ENGLISH)
        finally:
            QLocale.setDefault(old)

    def test_saved_values_roundtrip_and_unsupported_values_fall_back(self):
        serializer = LanguageSerializer()
        for language in Language:
            self.assertIs(serializer.deserialize(serializer.serialize(language)), language)
        for invalid in (None, [], {}, "fr_FR", "invalid-locale"):
            self.assertIs(serializer.deserialize(invalid), Language.AUTO)
        for traditional in ("zh_HK", "zh_MO", "zh_TW"):
            self.assertIs(serializer.deserialize(traditional), Language.CHINESE_TRADITIONAL)
        self.assertIs(serializer.deserialize("en_GB"), Language.ENGLISH)

    def test_system_locale_fallback_is_script_aware(self):
        for name in ("zh_TW", "zh_HK", "zh_MO"):
            self.assertEqual(locale_key(QLocale(name)), "zh_TW")
        self.assertEqual(locale_key(QLocale("zh_SG")), "zh_CN")
        self.assertEqual(locale_key(QLocale("fr_FR")), "en_US")


class SelectionTests(unittest.TestCase):
    def test_every_old_model_label_migrates_without_changing_identity(self):
        for group in CRISP_ASR_BACKENDS:
            ids = [model_id(m) for m in group["models"]]
            self.assertEqual(len(ids), len(set(ids)))
            for model in group["models"]:
                with self.subTest(group=group["backend"], model=model["value"]):
                    old = (group["label"], model["label"])
                    new = normalize_selection(*old, CRISP_ASR_VAD_METHODS[1]["label"],
                                              CRISP_ASR_LID_METHODS[0]["label"])
                    self.assertEqual(new, (group["backend"], model_id(model), "firered", "ecapa"))
                    self.assertEqual(resolve_model(*old), resolve_model(*new[:2]))
                    self.assertEqual(get_language_codes(*old), get_language_codes(*new[:2]))
                    self.assertEqual(resolve_vad_method(new[2]), "firered")
                    self.assertEqual(resolve_lid_method(new[3]), "ecapa")

    def test_unknown_values_never_fall_back_to_another_model(self):
        self.assertEqual(normalize_selection("missing", "missing", "missing", "missing"),
                         ("missing",) * 4)
        for backend, model in (("missing", ""), ("cohere", "missing"), ("", "removed-model")):
            with self.assertRaises(ValueError):
                resolve_model(backend, model)
        for resolver in (resolve_lid_method, resolve_vad_method):
            with self.assertRaises(ValueError):
                resolver("missing")


class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.old_values = {}
        self.save_patch = patch.object(cfg, "save")
        self.save_patch.start()
        self.addCleanup(self.save_patch.stop)
        # Restore the in-memory settings changed by widgets; disk writes mocked.
        for name in ("crisp_asr_backend", "crisp_asr_model", "crisp_asr_vad_method",
                     "crisp_asr_lid_method", "transcribe_language"):
            self.old_values[name] = getattr(cfg, name).value
        self.addCleanup(self.restore)

    def restore(self):
        for name, value in self.old_values.items():
            getattr(cfg, name).value = value

    def install(self, locale):
        translator = SupplementalTranslator(QLocale(locale), RESOURCE, self.app)
        self.app.installTranslator(translator)
        self.addCleanup(lambda: self.app.removeTranslator(translator))
        return translator

    def test_resources_cover_all_dynamic_labels(self):
        messages = json.loads(RESOURCE.read_text(encoding="utf-8"))
        sources = set(ENGINE_FLAVORS.values()) | set(ALIGNMENT_MODES.values())
        sources.update(m["label"] for m in ALIGNERS.values())
        sources.update(m.notice for m in MODEL_LICENSES.values())
        for group in CRISP_ASR_BACKENDS:
            sources.add(group["label"])
            sources.update(m["label"] for m in group["models"])
        sources.update(m["label"] for m in CRISP_ASR_LID_METHODS + CRISP_ASR_VAD_METHODS)
        for locale in ("en_US", "zh_TW"):
            self.assertFalse(sources - messages[locale].keys(), sources - messages[locale].keys())

    def test_missing_supplemental_entry_falls_through_to_existing_translator(self):
        class ExistingTranslator(QTranslator):
            def isEmpty(self):
                return False

            def translate(self, context, sourceText, disambiguation=None, n=-1):
                return "existing translation" if sourceText == "fallback-probe" else None

        existing = ExistingTranslator(self.app)
        self.app.installTranslator(existing)
        self.addCleanup(lambda: self.app.removeTranslator(existing))
        self.install("en_US")
        self.assertEqual(ui_text("fallback-probe"), "existing translation")

    def test_localized_widget_preserves_model_selection_across_restart(self):
        from app.components.CrispASRSettingWidget import CrispASRSettingWidget
        group = next(b for b in CRISP_ASR_BACKENDS if b["backend"] == "parakeet")
        model = next(m for m in group["models"] if m["value"] == "reazonspeech")
        cfg.crisp_asr_backend.value = group["label"]
        cfg.crisp_asr_model.value = model["label"]
        for locale in ("en_US", "zh_TW", "zh_CN"):
            translator = self.install(locale)
            widget = CrispASRSettingWidget()
            self.assertEqual(widget.backend_card.current_key(), "parakeet")
            self.assertEqual(widget.model_card.current_key(), "parakeet::reazonspeech")
            self.assertEqual(cfg.crisp_asr_model.value, "parakeet::reazonspeech")
            self.assertEqual(resolve_model(widget.backend_card.current_key(), widget.model_card.current_key()),
                             ("parakeet", "reazonspeech"))
            self.assertEqual(widget.model_card.comboBox.currentText(), ui_text(model["label"]))
            widget.deleteLater()
            self.app.removeTranslator(translator)

    def test_unknown_saved_selection_stays_invalid_in_widget(self):
        from app.components.CrispASRSettingWidget import CrispASRSettingWidget
        cfg.crisp_asr_backend.value = "cohere"
        cfg.crisp_asr_model.value = "removed-model"
        self.install("en_US")
        widget = CrispASRSettingWidget()
        self.assertEqual(cfg.crisp_asr_model.value, "removed-model")
        self.assertFalse(widget.license_card.isEnabled())
        widget.deleteLater()

    def test_license_dialog_text_is_localized_not_license_identity(self):
        from app.components import crisp_asr_license_dialog as gate
        for locale in ("en_US", "zh_TW"):
            translator = self.install(locale)
            config = TranscribeConfig(transcribe_model=TranscribeModelEnum.CRISP_ASR,
                                     crisp_asr_backend="sensevoice", crisp_asr_model="sensevoice::auto")
            fake = MagicMock()
            fake.crisp_asr_license_acceptances.value = []
            with patch.object(gate, "cfg", fake), patch.object(gate, "MessageBox") as dialog:
                dialog.return_value.exec.return_value = False
                self.assertFalse(gate.ensure_crispasr_license(MagicMock(), config))
                self.assertEqual(dialog.call_args.args[0], ui_text("确认模型授权"))
                self.assertIn("sensevoice / auto", dialog.call_args.args[1])
                fake.set.assert_not_called()
            self.app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()