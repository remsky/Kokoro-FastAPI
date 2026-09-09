import asyncio
import glob
import io
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from api.src.core import paths
from api.src.core.config import settings
from api.src.inference import inno_tuner
from api.src.inference.base import AudioChunk
from api.src.inference.voice_manager import VoiceManager
from api.src.main import app
from api.src.routers import openai_compatible
from api.src.routers.openai_compatible import get_tts_service

client = TestClient(app)


def wav_bytes(seconds=4.0, sr=24000, channels=1, fmt="WAV"):
    buf = io.BytesIO()
    frames = np.zeros((int(seconds * sr), channels), dtype=np.float32)
    sf.write(buf, frames, sr, format=fmt)
    return buf.getvalue()


def post(data=None, seconds=4.0, raw=None):
    return client.post(
        "/dev/tune",
        files={"audio": ("ref.wav", raw or wav_bytes(seconds), "audio/wav")},
        data=data or {},
    )


def transient():
    return VoiceManager._instance._transient


def temp_packs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "a_tune_*.pt")))


@pytest.fixture
def fake_tuner(monkeypatch):
    monkeypatch.setattr(settings, "enable_inno_tuner", True)
    monkeypatch.setattr(settings, "allow_local_voice_saving", True)
    monkeypatch.setattr(inno_tuner, "_tuner", object())
    calls = []

    def enroll(wav, sr, tuner, fmax=None, head=True):
        if len(wav) / sr < 3:
            raise ValueError("reference is too short; need at least 3 s")
        calls.append((round(len(wav) / sr, 3), head, fmax))
        return torch.zeros(510, 1, 256), {"af_heart": 1.0}

    import inno_kokoro.enroll

    monkeypatch.setattr(inno_kokoro.enroll, "enroll", enroll)
    return calls


@pytest.fixture
def service(fake_tuner, monkeypatch):
    seen = []

    async def gen(**kwargs):
        seen.append(kwargs)
        yield AudioChunk(np.zeros(1, dtype=np.int16), output=b"abc")

    svc = AsyncMock()
    svc.generate_audio_stream = gen
    monkeypatch.setattr(VoiceManager, "_instance", VoiceManager())
    svc.model_manager = MagicMock()
    svc.model_manager.get_backend.return_value.forget_voice = MagicMock()
    app.dependency_overrides[get_tts_service] = lambda: svc
    monkeypatch.setattr(
        openai_compatible, "get_tts_service", AsyncMock(return_value=svc)
    )
    yield svc, seen
    app.dependency_overrides.pop(get_tts_service, None)


def test_503_when_tuner_missing(monkeypatch):
    monkeypatch.setattr(settings, "enable_inno_tuner", True)
    monkeypatch.setattr(inno_tuner, "_tuner", None)
    assert post({"return_voice_pack": "true"}).status_code == 503


def test_403_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_inno_tuner", False)
    assert post({"return_voice_pack": "true"}).status_code == 403


def test_return_voice_pack_and_cleanup(service, fake_tuner):
    before = temp_packs()
    r = post({"return_voice_pack": "true", "prosody_head": "false"})
    assert r.status_code == 200
    assert fake_tuner[-1][1] is False
    assert r.headers["content-disposition"].endswith('filename="a_tune.pt"')
    assert torch.load(io.BytesIO(r.content), weights_only=True).shape == (510, 1, 256)
    assert temp_packs() == before
    assert transient() == {}


def test_403_when_voice_saving_disabled(service, monkeypatch):
    monkeypatch.setattr(settings, "allow_local_voice_saving", False)
    assert post({"save_voice": "am_me"}).status_code == 403
    assert post({"return_voice_pack": "true"}).status_code == 200


def test_save_voice_writes_voices_dir(service, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "voices_dir", str(tmp_path))
    before = temp_packs()
    r = post({"save_voice": "AM_Me"})
    assert r.status_code == 200
    assert r.json() == {"voice": "am_me_tuned"}
    assert torch.load(tmp_path / "am_me_tuned.pt", weights_only=True).shape == (
        510,
        1,
        256,
    )
    assert post({"save_voice": "am_me"}).status_code == 409
    assert post({"save_voice": "am_me_tuned"}).status_code == 409
    assert post({"save_voice": "B_Me"}).json() == {"voice": "b_me_tuned"}
    assert post({"save_voice": "ax_me"}).json() == {"voice": "ax_me_tuned"}
    for bad in (
        "me",
        "am-me",
        "am_me!",
        "am__me",
        "am_me_",
        "xf_me",
        "am_",
        "a_",
        "ax_",
        "a1_me",
    ):
        assert post({"save_voice": bad}).status_code == 400, bad
    assert temp_packs() == before
    assert asyncio.run(paths.list_voices()) == [
        "am_me_tuned",
        "ax_me_tuned",
        "b_me_tuned",
    ]
    assert asyncio.run(VoiceManager().get_voice_path("am_me_tuned")) == str(
        tmp_path / "am_me_tuned.pt"
    )


def test_knobs_and_size_cap(service, fake_tuner, monkeypatch):
    assert post({"return_voice_pack": "true", "fmax": "300"}).status_code == 200
    assert fake_tuner[-1] == (4.0, True, 300.0)
    assert post({"return_voice_pack": "true", "fmax": "5"}).status_code == 422
    monkeypatch.setattr(inno_tuner, "MAX_UPLOAD_BYTES", 1000)
    assert post({"return_voice_pack": "true"}).status_code == 413


def test_decode_is_bounded(service, fake_tuner):
    long_flac = wav_bytes(seconds=120, fmt="FLAC")
    assert len(long_flac) < inno_tuner.MAX_UPLOAD_BYTES
    assert post({"return_voice_pack": "true"}, raw=long_flac).status_code == 200
    assert fake_tuner[-1][0] == 30.0
    for raw in (wav_bytes(channels=4), wav_bytes(sr=192000)):
        assert post({"return_voice_pack": "true"}, raw=raw).status_code == 400


def test_400_on_bad_input(service):
    assert post({}).status_code == 400
    assert post({"return_voice_pack": "true"}, raw=b"not audio").status_code == 400
    assert post({"return_voice_pack": "true"}, seconds=2).status_code == 400
    assert post({"request": "{not json"}).status_code == 400
    assert (
        post({"request": json.dumps({"input": "hi", "speed": 99})}).status_code == 400
    )


def test_speech_uses_transient_voice_then_forgets_it(service):
    svc, seen = service
    before = temp_packs()
    r = post(
        {
            "request": json.dumps(
                {"input": "hello", "response_format": "wav", "allow_voice_tags": True}
            )
        }
    )
    assert r.status_code == 200
    assert r.content == b"abc"
    assert r.headers["content-type"] == "audio/wav"
    voice = seen[0]["voice"]
    assert voice.startswith("a_tune_")
    assert seen[0]["allow_voice_tags"] is False
    assert temp_packs() == before
    assert transient() == {}
    svc.model_manager.get_backend.return_value.forget_voice.assert_called_once()
    assert svc.model_manager.get_backend.return_value.forget_voice.call_args[0][
        0
    ].endswith(f"{voice}.pt")


def test_whole_response_with_download_link(service, monkeypatch, tmp_path):
    svc, _ = service
    monkeypatch.setattr(settings, "temp_file_dir", str(tmp_path))
    svc.generate_audio.return_value = AudioChunk(
        np.zeros(1, dtype=np.int16), output=b"abc"
    )
    before = temp_packs()
    r = post(
        {
            "request": json.dumps(
                {"input": "hello", "stream": False, "return_download_link": True}
            )
        }
    )
    assert r.status_code == 200
    assert r.content == b"abc"
    assert svc.generate_audio.call_args.kwargs["voice"].startswith("a_tune_")
    download = tmp_path / os.path.basename(r.headers["X-Download-Path"])
    assert download.read_bytes() == b"abc"
    assert temp_packs() == before
    assert transient() == {}
    svc.model_manager.get_backend.return_value.forget_voice.assert_called_once()


def test_stream_failure_still_discards_pack(service):
    svc, _ = service

    async def boom(**kwargs):
        yield AudioChunk(np.zeros(1, dtype=np.int16), output=b"abc")
        raise RuntimeError("inference fell over")

    svc.generate_audio_stream = boom
    before = temp_packs()
    with pytest.raises(RuntimeError):
        post({"request": json.dumps({"input": "hello", "stream": True})})
    assert temp_packs() == before
    assert transient() == {}
    svc.model_manager.get_backend.return_value.forget_voice.assert_called_once()


def test_enrollment_is_serialized(fake_tuner, monkeypatch):
    import threading
    import time

    import inno_kokoro.enroll

    in_flight, peak = [0], [0]

    def slow_enroll(wav, sr, tuner, fmax=None, head=True):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        in_flight[0] -= 1
        return torch.zeros(510, 1, 256), {}

    monkeypatch.setattr(inno_kokoro.enroll, "enroll", slow_enroll)
    data = wav_bytes(4.0)
    paths = []
    threads = [
        threading.Thread(target=lambda: paths.append(inno_tuner.tune(data)))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for path in paths:
        os.remove(path)
    assert peak[0] == 1


def test_forget_voice_evicts_pipeline_cache(tmp_path):
    from types import SimpleNamespace

    from api.src.inference.kokoro_v1 import KokoroV1

    backend = KokoroV1()
    pack = str(tmp_path / "a_tune_x.pt")
    temp_copy = os.path.join(tempfile.gettempdir(), "temp_voice_a_tune_x.pt")
    backend._voice_cache[f"{pack}:cpu"] = torch.zeros(1)
    backend._pipelines["a"] = SimpleNamespace(voices={temp_copy: torch.zeros(1)})
    backend.forget_voice(pack)
    assert backend._voice_cache == {}
    assert backend._pipelines["a"].voices == {}


@pytest.mark.asyncio
async def test_transient_voice_resolves_before_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "voices_dir", str(tmp_path))
    manager = VoiceManager()
    manager.register_transient("a_tune_x", str(tmp_path / "x.pt"))
    assert await manager.get_voice_path("a_tune_x") == str(tmp_path / "x.pt")
    manager.forget_transient("a_tune_x")
    with pytest.raises(FileNotFoundError):
        await manager.get_voice_path("a_tune_x")


def test_load_puts_tuner_on_configured_device(monkeypatch):
    import inno_kokoro.enroll

    built = []

    class Tuner:
        version = "0.2.0"

        def __init__(self, path, device):
            built.append((path, device))
            self.device = device

    monkeypatch.setattr(inno_kokoro.enroll, "Tuner", Tuner)
    monkeypatch.setattr(inno_tuner, "_tuner", None)
    monkeypatch.setattr(settings, "use_gpu", False)
    inno_tuner.load()
    assert inno_tuner.available()
    assert built == [(inno_tuner.weights_path(), "cpu")]
    assert inno_tuner.weights_path().endswith(
        os.path.join("v1_0", "inno_tuner", "model.safetensors")
    )


def test_tune_without_tuner_raises(monkeypatch):
    monkeypatch.setattr(inno_tuner, "_tuner", None)
    with pytest.raises(RuntimeError):
        inno_tuner.tune(wav_bytes())


def test_stereo_clip_is_mixed_to_mono(service, monkeypatch):
    import inno_kokoro.enroll

    shapes = []

    def enroll(wav, sr, tuner, fmax=None, head=True):
        shapes.append(tuple(wav.shape))
        return torch.zeros(510, 1, 256), {}

    monkeypatch.setattr(inno_kokoro.enroll, "enroll", enroll)
    r = post({"return_voice_pack": "true"}, raw=wav_bytes(channels=2))
    assert r.status_code == 200
    assert shapes == [(4 * 24000,)]


def test_speech_failure_before_streaming_discards_pack(service):
    svc, _ = service
    svc.generate_audio.side_effect = RuntimeError("inference fell over")
    before = temp_packs()
    r = post({"request": json.dumps({"input": "hello", "stream": False})})
    assert r.status_code == 500
    assert temp_packs() == before
    assert transient() == {}
    svc.model_manager.get_backend.return_value.forget_voice.assert_called_once()


@pytest.mark.asyncio
async def test_discard_never_raises(monkeypatch):
    from api.src.routers.tune import _discard

    monkeypatch.setattr(VoiceManager, "_instance", VoiceManager())
    svc = MagicMock()
    svc.model_manager.get_backend.return_value.forget_voice.side_effect = RuntimeError(
        "no pipeline"
    )
    await _discard(
        svc, "a_tune_x", os.path.join(tempfile.gettempdir(), "a_tune_gone.pt")
    )


def test_startup_survives_tuner_load_failure(monkeypatch):
    from api.src.inference.model_manager import ModelManager

    monkeypatch.setattr(settings, "enable_inno_tuner", True)
    monkeypatch.setattr(inno_tuner, "_tuner", None)
    monkeypatch.setattr(
        inno_tuner, "load", MagicMock(side_effect=OSError("no weights"))
    )
    monkeypatch.setattr(ModelManager, "_instance", None)
    monkeypatch.setattr(VoiceManager, "_instance", None)
    monkeypatch.setattr(
        ModelManager,
        "initialize_with_warmup",
        AsyncMock(return_value=("cpu", "kokoro_v1", 1)),
    )
    with TestClient(app) as booted:
        inno_tuner.load.assert_called_once()
        assert not inno_tuner.available()
        assert (
            booted.post(
                "/dev/tune", files={"audio": ("ref.wav", wav_bytes(), "audio/wav")}
            ).status_code
            == 503
        )
