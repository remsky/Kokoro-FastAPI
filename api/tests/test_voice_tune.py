"""Tests for the voice tune pack layout, the KModel hook plumbing and POST /dev/voices/tune."""

import asyncio
import base64
import io
import pathlib
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from api.src.core.config import settings
from api.src.inference import kokoro_v1, tune_adapter, voice_tune
from api.src.main import app
from api.src.routers.development import get_tts_service
from api.src.services.tts_service import TTSService

client = TestClient(app)
R, WIDTH = 128, 386  # the shipped adapter's r width and pack width


@contextmanager
def override_tts_service(service):
    async def _override():
        return service

    app.dependency_overrides[get_tts_service] = _override
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_tts_service, None)


def _wav_b64(seconds=2.0, sr=16000, hz=140.0):
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)).astype(
        np.float32
    )
    buf = io.BytesIO()
    sf.write(buf, tone * 0.3, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# prosody helpers and the KModel hook plumbing (tune_adapter is vendored 1:1)
# ---------------------------------------------------------------------------


def test_stats_tracks_a_harmonic_tone():
    wav, _ = voice_tune.decode_audio(base64.b64decode(_wav_b64(hz=220.0)))
    mean, std, _ = tune_adapter.stats(wav, 16000)
    assert mean == pytest.approx(13.7, abs=0.5)  # 220 Hz re 100 Hz, no octave drop
    assert std < 0.5


def test_rate_counts_amplitude_peaks():
    torch.manual_seed(0)
    t = torch.arange(16000 * 4) / 16000
    am = (0.3 + 0.7 * (0.5 + 0.5 * torch.cos(2 * torch.pi * 5 * t))) * torch.randn(
        len(t)
    )
    assert tune_adapter.rate(am, 16000) == pytest.approx(5, abs=0.7)


def _stand_in_model():
    """One AdaIN layer plus a decoder that returns its F0 argument, enough to see the hooks act."""
    from kokoro.istftnet import AdaIN1d

    class Decoder(torch.nn.Module):
        def forward(self, asr, f0, n, s):
            return f0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = AdaIN1d(256, 4)
            self.decoder = Decoder()
            self.device = torch.device("cpu")

        def forward(self, ps, ref_s, speed=1, return_output=False):
            self.norm.fc(ref_s)
            return self.decoder(
                None, torch.tensor([0.0, 100.0, 400.0]), None, ref_s
            ), speed

    return Model()


def test_hooks_split_wide_pack_and_keep_stock_untouched():
    model = _stand_in_model()
    slot, fc_rs = tune_adapter.attach(model, r_dim=R)
    assert slot == 0 and len(fc_rs) == 1 and tune_adapter.r_dim_of(model) == R
    with torch.no_grad():
        fc_rs[0].weight.fill_(1.0)
    seen = {}
    model.norm.fc.register_forward_hook(
        lambda m, a, o: seen.update(r=tune_adapter.residual._state.r)
    )
    tune_adapter.hook_model(model)

    wide = torch.zeros(1, WIDTH)
    wide[0, 256:384] = 0.5
    wide[0, 384], wide[0, 385] = 0.0, 0.5
    f0, speed = model("x", wide, 2.0, return_output=True)
    assert seen["r"].eq(0.5).all()
    assert speed == pytest.approx(1.0)  # request speed multiplies the enrolled one
    assert f0[0] == 0 and (12 * torch.log2(f0[1:] / 100)).mean().abs() < 1e-4

    stock = torch.randn(1, tune_adapter.STYLE)
    f0, speed = model("x", stock, 1.25, return_output=True)
    assert seen["r"] is None and speed == 1.25 and f0[2] == 400


def test_hooks_reject_other_widths():
    model = _stand_in_model()
    tune_adapter.attach(model, r_dim=R)
    tune_adapter.hook_model(model)
    with pytest.raises(RuntimeError):
        model("x", torch.zeros(1, 300))


# ---------------------------------------------------------------------------
# POST /dev/voices/tune
# ---------------------------------------------------------------------------


def test_adapter_path_prefers_model_dir_then_hf(monkeypatch):
    backend = kokoro_v1.KokoroV1.__new__(kokoro_v1.KokoroV1)
    monkeypatch.setattr(
        kokoro_v1.paths, "get_model_path", AsyncMock(return_value="/models/a.st")
    )
    assert asyncio.run(backend._adapter_path("v1_0/a.st")) == "/models/a.st"
    monkeypatch.setattr(
        kokoro_v1.paths, "get_model_path", AsyncMock(side_effect=FileNotFoundError)
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda repo, fn: f"hf:{repo}/{fn}"
    )
    assert (
        asyncio.run(backend._adapter_path("Remsky/kokoro-tune-adapter"))
        == "hf:Remsky/kokoro-tune-adapter/model.safetensors"
    )
    with pytest.raises(FileNotFoundError):  # a missing local file never hits HF
        asyncio.run(backend._adapter_path("v1_0/missing.safetensors"))


def _mock_service(backend=None):
    service = MagicMock(spec=TTSService)
    manager = MagicMock()
    manager.ensure_backend = AsyncMock()
    manager.get_backend = MagicMock(return_value=backend)
    service.model_manager = manager
    return service


def test_tune_endpoint_403_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "tune_adapter", None)
    backend = MagicMock()
    with override_tts_service(_mock_service(backend)):
        response = client.post(
            "/dev/voices/tune", json={"name": "af_test", "audio": _wav_b64()}
        )
    assert response.status_code == 403
    backend.enroll_voice.assert_not_called()


def test_tune_endpoint_rejects_bad_names(monkeypatch):
    monkeypatch.setattr(settings, "tune_adapter", "adapter.safetensors")
    with override_tts_service(_mock_service(MagicMock())):
        for name in ("../evil", "af_john-doe"):  # a hyphen would parse as a blend subtraction
            response = client.post(
                "/dev/voices/tune", json={"name": name, "audio": _wav_b64()}
            )
            assert response.status_code == 422, name


def test_tune_endpoint_400_on_undecodable_audio(monkeypatch):
    monkeypatch.setattr(settings, "tune_adapter", "adapter.safetensors")
    with override_tts_service(_mock_service(MagicMock())):
        response = client.post(
            "/dev/voices/tune",
            json={"name": "af_test", "audio": base64.b64encode(b"not audio").decode()},
        )
    assert response.status_code == 400


def test_tune_endpoint_saves_pack_and_evicts_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tune_adapter", "adapter.safetensors")
    monkeypatch.setattr(
        "api.src.core.paths.get_tune_voices_dir", lambda: str(tmp_path)
    )
    pack = torch.zeros(510, 1, WIDTH)
    pack[..., -2], pack[..., -1] = 2.5, 0.9
    backend = MagicMock()
    backend.enroll_voice = MagicMock(return_value=pack)
    with override_tts_service(_mock_service(backend)):
        response = client.post(
            "/dev/voices/tune",
            json={"name": "af_test", "audio": _wav_b64(), "strength": 1.5},
        )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "voice": "af_test",
        "adapter": "",
        "speed": 0.9,
        "f0_mean_st": 2.5,
    }
    saved = torch.load(tmp_path / "af_test.pt", weights_only=True)
    assert saved.shape == (510, 1, WIDTH)
    backend.evict_voice.assert_called_once_with(str(tmp_path / "af_test.pt"))
    wav, sr, strength = backend.enroll_voice.call_args.args
    assert sr == 16000 and wav.ndim == 1 and len(wav) == 32000 and strength == 1.5


def test_tune_endpoint_refuses_stock_voice_names(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tune_adapter", "adapter.safetensors")
    monkeypatch.setattr("api.src.core.paths.get_voices_dir", lambda: str(tmp_path))
    (tmp_path / "af_test.pt").write_bytes(b"")
    backend = MagicMock()
    with override_tts_service(_mock_service(backend)):
        response = client.post(
            "/dev/voices/tune", json={"name": "af_test", "audio": _wav_b64()}
        )
    assert response.status_code == 409
    backend.enroll_voice.assert_not_called()


def test_adapter_id_is_name_plus_weights_hash(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"w": torch.zeros(1)}, tmp_path / "a.safetensors", metadata={"name": "inno_v1"}
    )
    save_file(
        {"w": torch.ones(1)}, tmp_path / "b.safetensors", metadata={"name": "inno_v1"}
    )
    (alias, a), (_, b) = (
        voice_tune.adapter_ids(str(tmp_path / f))
        for f in ("a.safetensors", "b.safetensors")
    )
    assert alias == "inno_v1"
    assert a.startswith("inno_v1_") and len(a) == len("inno_v1_") + 8 and a != b


def test_voice_lookup_covers_stock_then_installed_adapter_dir(monkeypatch, tmp_path):
    stock = tmp_path / "stock"
    monkeypatch.setattr(settings, "voices_dir", str(stock))
    monkeypatch.setattr(settings, "tune_voices_dir", str(tmp_path))
    from api.src.core import paths

    monkeypatch.setattr(paths, "_tune", {})
    stock.mkdir()
    tune = pathlib.Path(paths.register_tune_adapter("inno_v1_abcd1234", "inno_v1"))
    assert tune == tmp_path / "inno_v1_abcd1234" and paths.get_tune_voices_dir() == str(
        tune
    )
    (stock / "af_stock.pt").write_bytes(b"")
    (tune / "af_mine.pt").write_bytes(b"")
    (tune / "af_stock.pt").write_bytes(b"")
    assert asyncio.run(paths.list_voices()) == ["af_mine", "af_stock"]
    assert asyncio.run(paths.get_voice_path("af_mine")) == str(tune / "af_mine.pt")
    assert asyncio.run(paths.get_voice_path("af_stock")) == str(stock / "af_stock.pt")
    assert paths.tune_alias() == "inno_v1"

    paths.clear_tune_adapter()
    assert paths.tune_alias() == ""
    assert asyncio.run(paths.list_voices()) == ["af_stock"]
    with pytest.raises(FileNotFoundError):
        asyncio.run(paths.get_voice_path("af_mine"))
