"""Offline regression tests: no model downloads or real settings writes."""

import importlib
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core.bk_asr.crisp_asr import CrispASR
from app.core.bk_asr.crisp_asr_catalog import CRISP_ASR_BACKENDS, resolve_model
from app.core.bk_asr.crisp_asr_licenses import (
    acceptance_key, accepted_license_tag, model_license,
)
from app.core.entities import BatchTaskType, TranscribeConfig, TranscribeModelEnum


def selection(backend, value="auto"):
    for group in CRISP_ASR_BACKENDS:
        for model in group["models"]:
            if model["backend"] == backend and model["value"] == value:
                return group["label"], model["label"]
    raise AssertionError((backend, value))


def configuration(backend="sensevoice", value="auto"):
    labels = selection(backend, value)
    return TranscribeConfig(
        transcribe_model=TranscribeModelEnum.CRISP_ASR,
        transcribe_language="en",
        crisp_asr_backend=labels[0], crisp_asr_model=labels[1],
    )


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "test.wav"
        self.audio.touch()
        self.model_patch = patch("app.core.bk_asr.crisp_asr.MODEL_PATH", Path(self.temp.name))
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def command(self, **kwargs):
        asr = CrispASR(str(self.audio), **kwargs)
        return asr._build_command(self.audio, self.audio.with_suffix(""))

    def test_kotoba_preserves_exact_filename_and_repo(self):
        model = "kotoba-whisper-v2.2-ggml-q8_0.bin"
        cmd = self.command(model=model)
        self.assertEqual(cmd[cmd.index("-m") + 1], model)
        self.assertEqual(cmd[cmd.index("--hf-repo") + 1], "kenrouse/kotoba-whisper-v2.2-ggml")
        self.assertEqual(cmd[cmd.index("--hf-file") + 1], model)

    def test_missing_standard_whisper_models_do_not_become_base(self):
        for size in ("tiny", "base", "small", "medium", "large-v3"):
            with self.subTest(size=size):
                name = f"ggml-{size}.bin"
                cmd = self.command(model=name)
                self.assertEqual(cmd[cmd.index("-m") + 1], name)
                self.assertEqual(cmd[cmd.index("--hf-repo") + 1], "ggerganov/whisper.cpp")

    def test_exact_local_model_skips_download_override(self):
        name = "ggml-small.bin"
        local = Path(self.temp.name) / name
        local.touch()
        cmd = self.command(model=name)
        self.assertEqual(cmd[cmd.index("-m") + 1], str(local))
        self.assertNotIn("--hf-repo", cmd)

    def test_similar_filename_is_not_selected(self):
        (Path(self.temp.name) / "ggml-small-q5_0.bin").touch()
        cmd = self.command(model="ggml-small.bin")
        self.assertEqual(cmd[cmd.index("-m") + 1], "ggml-small.bin")
        self.assertIn("--hf-repo", cmd)

    def test_unknown_missing_whisper_file_fails(self):
        with self.assertRaises(ValueError):
            self.command(model="missing-custom.bin")

    def test_german_wav2vec2_routing(self):
        labels = selection("wav2vec2", "wav2vec2-de")
        self.assertEqual(resolve_model(*labels), ("wav2vec2", "wav2vec2-de"))

    def test_standalone_ctc_routes_to_ctc_backend(self):
        for value in ("parakeet-ctc-0.6b", "parakeet-ctc-1.1b"):
            labels = selection("fastconformer-ctc", value)
            self.assertEqual(resolve_model(*labels), ("fastconformer-ctc", value))

    def test_restricted_model_requires_explicit_approval(self):
        with self.assertRaises(ValueError):
            self.command(backend="sensevoice")

    def test_approved_model_emits_exact_tag(self):
        license = model_license("sensevoice")
        key = acceptance_key("sensevoice", "auto", license)
        cmd = self.command(backend="sensevoice", license_acceptances=(key,))
        self.assertEqual(cmd[cmd.index("--accept-license") + 1], "funasr-v1.1")
        self.assertNotIn("all", cmd)

    def test_permissive_model_has_no_license_override(self):
        self.assertNotIn("--accept-license", self.command())

    def test_transcribe_startup_keywords_match_constructor(self):
        module = importlib.import_module("app.core.bk_asr.transcribe")
        config = configuration("cohere")
        config.crisp_asr_cohere_multilingual = True  # legacy settings remain readable
        captured = {}

        def create(audio, **kwargs):
            inspect.signature(CrispASR).bind(audio, **kwargs)
            captured.update(kwargs)
            return MagicMock()

        with patch.object(module, "CrispASR", side_effect=create):
            module.transcribe(str(self.audio), config)
        self.assertNotIn("cohere_multilingual", captured)
        self.assertEqual(captured["backend"], "cohere")


class LicenseTests(unittest.TestCase):
    def test_every_restricted_catalog_variant_is_covered(self):
        for backend in ("sensevoice", "paraformer", "funasr", "fun-asr-mlt-nano",
                        "gemma4-e2b", "moonshine-de", "moonshine-tiny-de"):
            for group in CRISP_ASR_BACKENDS:
                for model in group["models"]:
                    if model["backend"] == backend:
                        self.assertIsNotNone(model_license(backend, model["value"]))

    def test_approval_does_not_transfer_to_another_model_or_revision(self):
        key = acceptance_key("sensevoice", "auto", model_license("sensevoice"))
        for backend, model, keys in (
            ("paraformer", "auto", (key,)),
            ("sensevoice", "sensevoice-small-q8_0.gguf", (key,)),
            ("sensevoice", "auto", (key.replace("0.8.34", "0.8.25"),)),
            ("sensevoice", "auto", ("all",)),
            ("sensevoice", "auto", key),
        ):
            with self.subTest(backend=backend, model=model, keys=keys):
                with self.assertRaises(ValueError):
                    accepted_license_tag(backend, model, keys)


class ConsentDialogTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("app.components.crisp_asr_license_dialog")
        self.cfg = MagicMock()
        self.cfg.crisp_asr_license_acceptances.value = []
        self.cfg_patch = patch.object(self.module, "cfg", self.cfg)
        self.cfg_patch.start()
        self.addCleanup(self.cfg_patch.stop)
        self.dialog_patch = patch.object(self.module, "MessageBox")
        self.dialog = self.dialog_patch.start()
        self.addCleanup(self.dialog_patch.stop)
        self.parent = MagicMock()

    def test_cancel_does_not_persist_or_approve(self):
        self.dialog.return_value.exec.return_value = False
        config = configuration()
        self.assertFalse(self.module.ensure_crispasr_license(self.parent, config))
        self.cfg.set.assert_not_called()
        self.assertEqual(config.crisp_asr_license_acceptances, ())

    def test_accept_persists_exact_model_approval(self):
        self.dialog.return_value.exec.return_value = True
        config = configuration()
        self.assertTrue(self.module.ensure_crispasr_license(self.parent, config))
        key = acceptance_key("sensevoice", "auto", model_license("sensevoice"))
        self.cfg.set.assert_called_once_with(self.cfg.crisp_asr_license_acceptances, [key])
        self.assertEqual(config.crisp_asr_license_acceptances, (key,))

    def test_saved_confirmation_covers_restored_selection(self):
        key = acceptance_key("sensevoice", "auto", model_license("sensevoice"))
        self.cfg.crisp_asr_license_acceptances.value = [key]
        config = configuration()
        self.assertTrue(self.module.ensure_crispasr_license(self.parent, config))
        self.dialog.assert_not_called()
        self.assertEqual(config.crisp_asr_license_acceptances, (key,))

    def test_permissive_and_non_crisp_tasks_skip_dialog(self):
        self.assertTrue(self.module.ensure_crispasr_license(self.parent, configuration("cohere")))
        self.assertTrue(self.module.ensure_crispasr_license(self.parent, TranscribeConfig()))
        self.dialog.assert_not_called()


class LaunchTests(unittest.TestCase):
    def test_retry_preserves_pipeline_after_license_cancel(self):
        module = importlib.import_module("app.view.transcription_interface")
        original = SimpleNamespace(transcribe_config=configuration(), need_next_task=True)
        ui = SimpleNamespace(
            task=original, video_info=SimpleNamespace(file_path="a.wav"),
            transcription_interface=SimpleNamespace(is_processing=False), reset_ui=MagicMock(),
        )
        with patch("app.components.crisp_asr_license_dialog.ensure_crispasr_license", return_value=False), \
             patch.object(module.TaskFactory, "create_transcribe_task", return_value=original) as factory:
            self.assertFalse(module.VideoInfoCard.start_transcription(ui))
        factory.assert_called_once_with("a.wav", need_next_task=True)

    def test_single_cancel_does_not_start_thread_or_leave_processing_state(self):
        module = importlib.import_module("app.view.transcription_interface")
        ui = SimpleNamespace(
            task=SimpleNamespace(transcribe_config=configuration()),
            transcription_interface=SimpleNamespace(is_processing=False), reset_ui=MagicMock(),
        )
        with patch("app.components.crisp_asr_license_dialog.ensure_crispasr_license", return_value=False), \
             patch.object(module, "TranscriptThread") as thread:
            self.assertFalse(module.VideoInfoCard.start_transcription(ui, need_create_task=False))
        thread.assert_not_called()
        self.assertFalse(ui.transcription_interface.is_processing)

    def test_batch_cancel_enqueues_nothing_and_preserves_rows(self):
        module = importlib.import_module("app.view.batch_process_interface")
        ui = MagicMock()
        ui.task_type_combo.currentText.return_value = BatchTaskType.TRANSCRIBE.value
        with patch.object(module, "ensure_crispasr_license", return_value=False), \
             patch.object(module.TaskFactory, "create_transcribe_task"):
            result = module.BatchProcessInterface._submit_tasks(ui, ["a.wav", "b.wav"], restart=True)
        self.assertFalse(result)
        ui.batch_thread.add_task.assert_not_called()
        ui.task_table.rowCount.assert_not_called()

    def test_batch_worker_uses_approved_snapshot_in_all_transcription_modes(self):
        module = importlib.import_module("app.thread.batch_process_thread")
        for handler, task_type in (
            ("_handle_transcribe_task", BatchTaskType.TRANSCRIBE),
            ("_handle_trans_sub_task", BatchTaskType.TRANS_SUB),
            ("_handle_full_process_task", BatchTaskType.FULL_PROCESS),
        ):
            with self.subTest(handler=handler):
                snapshot = SimpleNamespace(transcribe_config=configuration())
                task = module.BatchTask("a.wav", task_type, snapshot)
                worker = MagicMock()
                with patch.object(module, "TranscriptThread") as thread:
                    getattr(module.BatchProcessThread, handler)(worker, task)
                    thread.assert_called_once_with(snapshot)
                worker.factory.create_transcribe_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()