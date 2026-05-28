"""Regression test for https://github.com/coqui-ai/TTS/issues/4415

When a pre-trained model (e.g. xtts_v2) is stored as a directory instead of a
single checkpoint file, ``manager.download_model()`` returns the directory path
as ``model_path`` and ``None`` as ``config_path``.  The server must detect this
case and pass ``model_dir=model_path`` to ``Synthesizer`` rather than
``tts_checkpoint=model_path, tts_config_path=None``, because the latter causes
``load_config(None)`` → ``TypeError: expected str, bytes or os.PathLike object,
not NoneType``.

Run standalone:  python tests/aux_tests/test_server_model_dir.py
Run via pytest (requires full TTS install):  pytest tests/aux_tests/test_server_model_dir.py
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stubs(model_dir: Path) -> dict:
    """Return a mapping of module-name → stub that lets server.py load without
    the full TTS dependency stack installed."""

    def _load_config(path):
        if path is None:
            raise TypeError(
                "expected str, bytes or os.PathLike object, not NoneType"
            )
        return {}

    tts_pkg = types.ModuleType("TTS")
    tts_pkg.__path__ = ["TTS"]
    tts_pkg.__package__ = "TTS"

    tts_server_pkg = types.ModuleType("TTS.server")
    tts_server_pkg.__path__ = ["TTS/server"]
    tts_server_pkg.__package__ = "TTS.server"

    tts_config_mod = types.ModuleType("TTS.config")
    tts_config_mod.load_config = _load_config

    class _FakeModelManager:
        def __init__(self, *a, **kw):
            pass

        def download_model(self, name):
            # Simulate xtts_v2: directory path, no config file path
            return str(model_dir), None, {"default_vocoder": None}

        def list_models(self):
            pass

    tts_utils_pkg = types.ModuleType("TTS.utils")
    tts_utils_pkg.__path__ = ["TTS/utils"]

    tts_manage_mod = types.ModuleType("TTS.utils.manage")
    tts_manage_mod.ModelManager = _FakeModelManager

    flask_mod = types.ModuleType("flask")
    flask_mod.Flask = mock.MagicMock(return_value=mock.MagicMock())
    flask_mod.render_template = mock.MagicMock()
    flask_mod.render_template_string = mock.MagicMock()
    flask_mod.request = mock.MagicMock()
    flask_mod.send_file = mock.MagicMock()

    return {
        "TTS": tts_pkg,
        "TTS.server": tts_server_pkg,
        "TTS.config": tts_config_mod,
        "TTS.utils": tts_utils_pkg,
        "TTS.utils.manage": tts_manage_mod,
        "flask": flask_mod,
    }


def _run_server_with_directory_model() -> dict:
    """Execute server.py (top-level module code) with stubs so that
    download_model returns a directory path for an xtts model.

    Returns the kwargs dict that was passed to Synthesizer.__init__.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "tts_models--multilingual--multi-dataset--xtts_v2"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        captured: dict = {}

        class _FakeTtsModel:
            num_speakers = 1
            num_languages = 1
            speaker_manager = None
            language_manager = None

        class _FakeTtsConfig:
            def get(self, key, default=None):
                return default

        class _CapturingSynthesizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tts_model = _FakeTtsModel()
                self.tts_config = _FakeTtsConfig()
                self.tts_speakers_file = kwargs.get("tts_speakers_file")
                self.tts_languages_file = kwargs.get("tts_languages_file")

        stubs = _build_stubs(model_dir)

        tts_synth_mod = types.ModuleType("TTS.utils.synthesizer")
        tts_synth_mod.Synthesizer = _CapturingSynthesizer
        stubs["TTS.utils.synthesizer"] = tts_synth_mod

        # Clear conflicting modules then inject stubs
        for k in list(sys.modules):
            if k.startswith("TTS") or k == "flask":
                sys.modules.pop(k, None)
        for k, v in stubs.items():
            sys.modules[k] = v

        server_key = "TTS.server.server"
        with mock.patch(
            "sys.argv",
            [
                "server.py",
                "--model_name",
                "tts_models/multilingual/multi-dataset/xtts_v2",
            ],
        ):
            loader = importlib.machinery.SourceFileLoader(
                server_key, "TTS/server/server.py"
            )
            spec = importlib.util.spec_from_loader(server_key, loader)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[server_key] = mod
            spec.loader.exec_module(mod)

        # Clean up stubs
        for k in list(sys.modules):
            if k.startswith("TTS") or k == "flask":
                sys.modules.pop(k, None)

        return captured


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_directory_model_uses_model_dir():
    """Synthesizer must receive ``model_dir=`` (not ``tts_checkpoint=``) when the
    downloaded model path is a directory and config_path is None.

    Without this fix the server crashes with:
        TypeError: expected str, bytes or os.PathLike object, not NoneType
    because ``load_config(None)`` is called inside ``_load_tts``.
    """
    kwargs = _run_server_with_directory_model()

    assert "model_dir" in kwargs, (
        "Synthesizer should be called with model_dir= for directory-based models "
        "(e.g. xtts_v2), but it was not.  tts_checkpoint was passed instead, "
        "which triggers the TypeError when tts_config_path is None."
    )
    assert not kwargs.get("tts_checkpoint"), (
        "tts_checkpoint must not be set when model_dir is used"
    )
    assert not kwargs.get("tts_config_path"), (
        "tts_config_path=None would cause load_config to crash"
    )


if __name__ == "__main__":
    test_directory_model_uses_model_dir()
    print("PASS: model_dir correctly passed to Synthesizer for directory-based model (xtts_v2)")
