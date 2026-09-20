import io
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import zipfile

from app.core.bk_asr.crisp_asr import CrispASR
from app.core.bk_asr.crisp_asr_catalog import CRISP_ASR_BACKENDS, CRISP_ASR_WHISPER_HF_REPOS, resolve_model
from app.core.bk_asr.crisp_asr_options import ALIGNERS, alignment_args
from app.thread import crisp_asr_download_thread as dl

ROOT = Path(__file__).resolve().parents[1]


class AlignmentTests(unittest.TestCase):
    def test_explicit_word_or_alignment_output_skips_timing_rewrite(self):
        from app.core.entities import TranscribeConfig, TranscribeModelEnum
        module = importlib.import_module("app.core.bk_asr.transcribe")
        for mode, words, optimize in (("default", False, True), ("default", True, False),
                                     ("force", False, False), ("native", False, False)):
            config = TranscribeConfig(transcribe_model=TranscribeModelEnum.CRISP_ASR,
                                     need_word_time_stamp=False,
                                     crisp_asr_alignment_mode=mode, crisp_asr_word_timestamps=words)
            asr = MagicMock()
            with patch.object(module, "CrispASR", return_value=asr):
                module.transcribe("test.wav", config)
            self.assertEqual(asr.run.return_value.optimize_timing.called, optimize)

    def test_default_retains_engine_policy(self):
        self.assertEqual(alignment_args("default", "unused", "auto", False), [])

    def test_native_disables_implicit_canary_aligner(self):
        self.assertEqual(alignment_args("native", "unused", "en", False), ["--no-auto-aligner"])

    def test_force_uses_correct_flag_and_real_word_contract(self):
        args = alignment_args("force", "wav2vec2-aligner-ja", "ja", True)
        self.assertIn("--force-aligner", args)
        self.assertNotIn("-fa", args)
        self.assertIn("--require-word-timestamps", args)
        self.assertEqual(args[args.index("--max-len") + 1], "1")

    def test_fallback_does_not_replace_existing_words(self):
        args = alignment_args("fallback", "qwen3-forced-aligner", "auto", False)
        self.assertIn("--aligner-model", args)
        self.assertNotIn("--force-aligner", args)

    def test_mismatched_unknown_and_auto_language_are_rejected(self):
        for mode, model, lang in (("bad", "auto", "en"), ("force", "native", "en"),
                                  ("force", "wav2vec2-aligner-ja", "zh"),
                                  ("force", "wav2vec2-aligner-ja", "auto")):
            with self.subTest(mode=mode, model=model, lang=lang), self.assertRaises(ValueError):
                alignment_args(mode, model, lang, False)

    def test_aligner_keys_exist_in_pinned_registry(self):
        text = (ROOT / "_build/CrispASR/src/crispasr_model_registry.cpp").read_text(encoding="utf-8")
        keys = set(re.findall(r'\{"([^"\n]+)",\s*"', text))
        self.assertTrue(set(ALIGNERS) <= keys)

    def test_command_and_gpu_off_select_cpu_only_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            wav = Path(temp) / "test.wav"
            wav.touch()
            asr = CrispASR(str(wav), engine_flavor="cuda13", use_gpu=False,
                           alignment_mode="force", word_timestamps=True)
            cmd = asr._build_command(wav, wav.with_suffix(""))
            self.assertIn("--no-gpu", cmd)
            self.assertIn("--require-word-timestamps", cmd)
            with patch("app.core.bk_asr.crisp_asr.CRISP_ASR_BIN", Path(temp) / "runtime/crispasr.exe"), \
                 patch.object(asr, "_download_engine") as download:
                asr._ensure_engine(lambda *_: None)
                self.assertEqual(download.call_args.kwargs["flavor"], "cpu")
                self.assertEqual(asr.crisp_asr_path.parent.name, "cpu")


class CatalogTests(unittest.TestCase):
    def test_catalog_models_exist_in_registry(self):
        source = (ROOT / "_build/CrispASR/src/crispasr_model_registry.cpp").read_text(encoding="utf-8")
        entries = re.findall(r'\{"([^"\n]+)",\s*"([^"\n]+)"', source)
        keys = {x[0] for x in entries}
        files = {x[1] for x in entries}
        for group in CRISP_ASR_BACKENDS:
            for model in group["models"]:
                backend, value = resolve_model(group["label"], model["label"])
                with self.subTest(backend=backend, value=value):
                    if value == "auto":
                        self.assertIn(backend, keys)
                    else:
                        self.assertTrue(value in keys or value in files or value in CRISP_ASR_WHISPER_HF_REPOS)

    def test_removed_tts_selection_cannot_silently_run_whisper(self):
        with self.assertRaises(ValueError):
            resolve_model("KugelAudio (多语言, ~14GB)", "auto")


class EngineTests(unittest.TestCase):
    def test_exact_full_assets_for_all_flavors(self):
        assets = [{"name": name, "browser_download_url": f"https://example.invalid/{name}"}
                  for name in dl.WINDOWS_ASSETS.values()]
        assets.insert(0, {"name": "crispasr-windows-x86_64-cuda-non-cuda.zip",
                          "browser_download_url": "wrong"})
        for flavor, name in dl.WINDOWS_ASSETS.items():
            with patch.object(dl.platform, "system", return_value="Windows"), \
                 patch.object(dl.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps({"assets": assets}).encode())):
                self.assertEqual(dl.get_release_asset_url(flavor=flavor), f"https://example.invalid/{name}")

    def test_missing_exact_asset_never_substitutes_flavor(self):
        with patch.object(dl.platform, "system", return_value="Windows"), \
             patch.object(dl.urllib.request, "urlopen", return_value=io.BytesIO(b'{"assets":[]}')), \
             self.assertRaises(RuntimeError):
            dl.get_release_asset_url(flavor="cuda13")

    def test_wrong_version_abi_and_failed_process_are_rejected(self):
        for version, abi, rc in (("0.8.25", "12", 0), ("0.8.34", "13", 0), ("0.8.34", "12", 1)):
            result = subprocess.CompletedProcess([], rc,
                f"version : {version}\nggml backends : cpu,cuda\ncuda runtime ABI: {abi}\n", "")
            with patch.object(dl.subprocess, "run", return_value=result), self.assertRaises(RuntimeError):
                dl.validate_engine("fake.exe", "cuda")

    def archive(self, root, unsafe=False):
        archive = root / "engine.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("package/crispasr.exe" if os.name == "nt" else "package/crispasr", b"new")
            if unsafe:
                zf.writestr("../outside.txt", b"bad")
        return archive

    def test_clean_install_removes_stale_dlls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "installed"
            target.mkdir()
            (target / "stale.dll").touch()
            archive = self.archive(root)
            with patch.object(dl, "validate_engine"):
                exe = dl._install_archive(archive, target, "cpu")
            self.assertEqual(Path(exe).read_bytes(), b"new")
            self.assertFalse((target / "stale.dll").exists())

    def test_invalid_archive_preserves_existing_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "installed"
            target.mkdir()
            sentinel = target / "keep.dll"
            sentinel.write_bytes(b"old")
            with patch.object(dl, "validate_engine", side_effect=RuntimeError("invalid")), self.assertRaises(RuntimeError):
                dl._install_archive(self.archive(root), target, "cpu")
            self.assertEqual(sentinel.read_bytes(), b"old")

    def test_unsafe_archive_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                dl._install_archive(self.archive(root, unsafe=True), root / "installed", "cpu")
            self.assertFalse((root / "outside.txt").exists())

    def test_missing_cuda_runtime_keeps_old_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "installed"
            target.mkdir()
            (target / "keep.dll").touch()
            with self.assertRaises(RuntimeError):
                dl._install_archive(self.archive(root), target, "cuda")
            self.assertTrue((target / "keep.dll").exists())


if __name__ == "__main__":
    unittest.main()