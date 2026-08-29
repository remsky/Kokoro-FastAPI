"""Kokoro-82M voice tune adapter, inference side.

install(model, "model.safetensors") once after building KModel. From then on a wide voice pack
[ROWS, 1, STYLE + r + 2] = (style STYLE | r | f0 mean st | speed) clones through the stock KPipeline,
with the r width read from the adapter file; stock [ROWS, 1, STYLE] packs are untouched.

Several adapters can share one model: each install() adds a slot and returns its index on heads.slot,
and use(slot) picks the one that renders next in this task. A pack is only meaningful to the adapter
that enrolled it, so the slot follows the pack, it is not a quality dial.
Self-check: python -m tune_adapter.residual  (stock voice bit-identical after install, wide pack renders)
"""

import threading
from contextvars import ContextVar

import torch
from kokoro.istftnet import AdaIN1d
from kokoro.modules import AdaLayerNorm
from safetensors import safe_open
from torch import nn

from .prosody import hz, st

ENC = 512  # speaker-encoder embedding dim (UniSpeech-SAT-sv)
STYLE = 256  # Kokoro style vector
ROWS = 510  # rows per pack, one per possible phoneme count
F0_VOICED_HZ = 50  # decoder F0 below this is treated as unvoiced

_state = (
    threading.local()
)  # r and f0 mean of the pack being rendered, set per forward by the KModel pre-hook
_slot_var = ContextVar(
    "tune_adapter_slot", default=0
)  # a ContextVar, not thread state: a server interleaves renders between chunks on one thread


def use(slot):
    """Pick which installed adapter renders from here on in this task; install() returns it as heads.slot.

    A pack carries an adapter's own style and r coordinates, so the slot has to be the one that enrolled
    it. Rendering a pack through another adapter is not a variation on the voice, it is a different one.
    """
    _slot_var.set(slot)


def _slot():
    return _slot_var.get()


class ResFC(nn.Module):
    """Wraps one adaptive-norm style projection: fc(s) + fc_r[slot](r). With r = None the output is stock.

    One fc_r per installed adapter, in install order.
    """

    def __init__(self, fc):
        super().__init__()
        self.fc = fc
        self.fc_r = nn.ModuleList()

    def add_slot(self, r_dim):
        """Add one adapter's residual projection, zeroed: a slot nobody loads weights into is a no-op."""
        fc_r = nn.Linear(
            r_dim, self.fc.out_features, bias=False, device=self.fc.weight.device
        )
        nn.init.zeros_(fc_r.weight)
        self.fc_r.append(fc_r)
        return fc_r

    def forward(self, s):
        h = self.fc(s)
        r = getattr(_state, "r", None)
        if r is None:
            return h
        return h + self.fc_r[_slot()](r)


def make_heads(r_dim):
    """Speaker embedding [1, ENC] -> style [1, STYLE] and r [1, r_dim]."""

    def mlp(out):
        return nn.Sequential(nn.Linear(ENC, ENC), nn.GELU(), nn.Linear(ENC, out))

    # tilt: the reference's measured spectral tilt (normalised scalar) moves the style along one learned direction.
    return nn.ModuleDict(
        {"style": mlp(STYLE), "r": mlp(r_dim), "tilt": nn.Linear(1, STYLE, bias=False)}
    )


def attach(model, scope="all", r_dim=128):
    """Wrap every AdaIN1d / AdaLayerNorm style projection in place and give this adapter a slot.

    Every wrapper in the model gets the slot, not just the ones in `scope`, so slot indices line up
    across adapters that were trained with different scopes; the out-of-scope ones keep their zero
    weights, which is exactly what "no residual here" means.
    Returns (slot, the fc_r layers this adapter loads into, in module order).
    """
    norms = [m for m in model.modules() if isinstance(m, (AdaIN1d, AdaLayerNorm))]
    for m in norms:
        if not isinstance(m.fc, ResFC):
            m.fc = ResFC(m.fc)
        m.fc.add_slot(r_dim)
    depths = {len(m.fc.fc_r) for m in norms}
    assert len(depths) == 1, (
        f"adaptive-norm layers disagree on slot count: {sorted(depths)}"
    )
    root = model.decoder if scope == "decoder" else model
    scoped = [m for m in root.modules() if isinstance(m, (AdaIN1d, AdaLayerNorm))]
    return depths.pop() - 1, [m.fc.fc_r[-1] for m in scoped]


def r_dim_of(model, slot=0):
    """r width of the adapter in `slot`."""
    return (
        next(m for m in model.modules() if isinstance(m, ResFC)).fc_r[slot].in_features
    )


def load_adapter(model, path):
    """attach + load weights. Returns the heads, used at enrollment only."""
    with safe_open(path, "pt", device=str(model.device)) as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}

    r_dim = sd["fc_r.0.weight"].shape[1]
    slot, fc_rs = attach(model, meta.get("scope", "all"), r_dim)
    n = sum(k.startswith("fc_r.") for k in sd)
    assert n == len(fc_rs), (
        f"{path}: {n} fc_r tensors vs {len(fc_rs)} adaptive-norm layers in this Kokoro"
    )
    with torch.no_grad():
        for i, m in enumerate(fc_rs):
            m.weight.copy_(sd[f"fc_r.{i}.weight"])

    heads = make_heads(r_dim).to(model.device)
    missing = heads.load_state_dict(
        {k[len("heads.") :]: v for k, v in sd.items() if k.startswith("heads.")},
        strict=False,
    ).missing_keys
    assert set(missing) <= {"tilt.weight"}, f"{path}: missing {missing}"
    if missing:  # adapter trained without the tilt loss: a zero direction leaves the style untouched
        with torch.no_grad():
            heads["tilt"].weight.zero_()
    heads.tilt_norm = (
        float(meta.get("tilt_mean", -7.8)),
        float(meta.get("tilt_sd", 1.8)),
    )  # z-score of the reference tilt, as in training
    heads.mu = {
        k: sd[f"mu.{k}"] for k in ("style", "r")
    }  # mean clone over training speakers, anchor for `strength`
    heads.slot = slot  # pass to use() to render this adapter's packs
    heads.name = meta.get("name", "adapter")
    return heads.eval().requires_grad_(False)


def hook_model(model):
    """The two hooks that make wide voice packs work through the stock KPipeline. Needs attach() first.

    Installing a second adapter does not hook again: the hooks read the current slot, not one adapter.
    """
    if getattr(model, "_tune_hooked", False):
        return
    model._tune_hooked = True

    def unpack(m, args, kwargs):
        # KPipeline.infer calls model(ps, pack[len(ps) - 1], speed, return_output=True)
        ps, ref_s, *rest = args
        speed = kwargs.pop("speed", rest[0] if rest else 1)
        if ref_s.shape[-1] == STYLE:  # stock voice
            _state.r = _state.f0 = None
            return (ps, ref_s, speed, *rest[1:]), kwargs
        style, r, f0_mean, enrolled_speed = torch.split(
            ref_s, [STYLE, r_dim_of(model, _slot()), 1, 1], dim=-1
        )
        _state.r = r
        _state.f0 = f0_mean.item()
        return (ps, style, speed * enrolled_speed.item(), *rest[1:]), kwargs

    def f0_shift(m, args):
        # Decoder args are (asr, F0, N, s). Move the voiced log-F0 mean onto the enrolled one.
        f0_mean = getattr(_state, "f0", None)
        if f0_mean is None:
            return None
        f0 = args[1].clone()
        voiced = f0 > F0_VOICED_HZ
        semitones = st(f0[voiced])
        f0[voiced] = hz(semitones + f0_mean - semitones.mean())
        return (args[0], f0, *args[2:])

    model.register_forward_pre_hook(unpack, with_kwargs=True)
    model.decoder.register_forward_pre_hook(f0_shift)


def install(model, path):
    """load_adapter + hook_model. Returns the heads; heads.slot is what use() takes."""
    heads = load_adapter(model, path)
    hook_model(model)
    return heads


def _self_check():
    import os

    from kokoro import KModel, KPipeline

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel().to(dev).eval()
    pipe = KPipeline(lang_code="a", model=model, device=dev)

    def render(voice):
        torch.manual_seed(0)  # the decoder adds noise, so equality needs a fixed seed
        return next(pipe("Hello there, this is a test.", voice=voice)).audio

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "model.safetensors"
    )
    before = render("af_heart").clone()
    heads = install(model, path)
    assert heads.slot == 0
    assert torch.equal(before, render("af_heart")), "stock voice changed after install"

    def pack_for(heads):
        e = torch.nn.functional.normalize(torch.randn(1, ENC, device=dev), dim=-1)
        row = torch.cat(
            [heads["style"](e), heads["r"](e), torch.tensor([[0.0, 1.0]], device=dev)],
            -1,
        )
        assert row.shape[-1] == STYLE + r_dim_of(model, heads.slot) + 2
        return (
            row[None].repeat(ROWS, 1, 1).cpu()
        )  # KPipeline takes a CPU tensor or a .pt path

    torch.manual_seed(1)
    pack = pack_for(heads)
    wav = render(pack)
    assert wav.dim() == 1 and wav.abs().max() > 0.05 and torch.isfinite(wav).all()
    assert torch.equal(before, render("af_heart")), (
        "stock voice changed after a clone render"
    )

    # A slot is an independent adapter: same weights in slot 1 must render slot 0's voice, and an empty slot none of it.
    twin = install(model, path)
    assert twin.slot == 1
    empty = attach(model, "all", r_dim_of(model))[0]
    use(twin.slot)
    torch.manual_seed(1)
    assert torch.equal(wav, render(pack_for(twin))), (
        "slot 1 differs from slot 0 on the same weights"
    )
    use(heads.slot)
    assert torch.equal(wav, render(pack)), "slot 0 changed after a second install"
    use(empty)
    assert not torch.equal(wav, render(pack)), "zeroed slot still applied a residual"
    use(heads.slot)
    assert torch.equal(before, render("af_heart")), (
        "stock voice changed after the slot switches"
    )
    print(
        "ok: stock bit-identical, wide pack renders",
        tuple(wav.shape),
        "slots",
        empty + 1,
    )


if __name__ == "__main__":
    _self_check()
