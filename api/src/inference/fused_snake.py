"""Opt-in FP32 Snake activation fused into one CUDA kernel.

Validated on Quadro K620 (sm_50) with exact waveform match vs stock Snake.
Requires CUDA + NVRTC; no CPU fallback.
"""

from __future__ import annotations

import ctypes as C
import glob
import os
from pathlib import Path
import types

import torch


def _find_nvrtc() -> str:
    env = os.environ.get("KOKORO_NVRTC_LIB")
    if env and Path(env).is_file():
        return env
    patterns = [
        "/usr/local/cuda/lib64/libnvrtc.so*",
        "/usr/lib/libnvrtc.so*",
        "/usr/lib64/libnvrtc.so*",
        str(Path(torch.__file__).resolve().parent / "lib" / "libnvrtc.so*"),
        str(
            Path.home()
            / ".local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*"
        ),
        "/usr/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    ]
    hits: list[str] = []
    for pattern in patterns:
        hits.extend(glob.glob(pattern))
    # Prefer versioned .so.N over bare .so when both exist
    hits = sorted(set(hits), key=lambda p: (0 if ".so." in Path(p).name else 1, p))
    if not hits:
        raise RuntimeError(
            "libnvrtc not found; set KOKORO_NVRTC_LIB or install CUDA/NVRTC"
        )
    return hits[0]


class Snake:
    """Compile and launch a contiguous FP32 Snake kernel on the current stream."""

    def __init__(self, arch: str | None = None):
        major, minor = torch.cuda.get_device_capability()
        arch = arch or f"compute_{major}{minor}"
        nv = C.CDLL(_find_nvrtc())
        self.driver = C.CDLL("libcuda.so.1")

        def bind(lib, name, args):
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = C.c_int
            return fn

        void = C.c_void_p
        ptr = C.POINTER(void)
        create = bind(
            nv,
            "nvrtcCreateProgram",
            [ptr, C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)],
        )
        compile_ = bind(nv, "nvrtcCompileProgram", [void, C.c_int, C.POINTER(C.c_char_p)])
        logsize = bind(nv, "nvrtcGetProgramLogSize", [void, C.POINTER(C.c_size_t)])
        getlog = bind(nv, "nvrtcGetProgramLog", [void, void])
        getsize = bind(nv, "nvrtcGetPTXSize", [void, C.POINTER(C.c_size_t)])
        getptx = bind(nv, "nvrtcGetPTX", [void, void])
        destroy = bind(nv, "nvrtcDestroyProgram", [ptr])

        source = b"""extern "C" __global__ void snake(
            const float* x, const float* a, float* y, int n, int channels, int length) {
          int i = blockIdx.x * blockDim.x + threadIdx.x;
          if (i >= n) return;
          float alpha = a[(i / length) % channels];
          float v = x[i];
          float s = sinf(__fmul_rn(alpha, v));
          y[i] = __fadd_rn(v, __fmul_rn(__frcp_rn(alpha), __fmul_rn(s, s)));
        }"""
        prog = void()
        self.check(create(C.byref(prog), source, b"snake.cu", 0, None, None))
        options = (C.c_char_p * 2)(f"--gpu-architecture={arch}".encode(), b"--std=c++11")
        err = compile_(prog, 2, options)
        size = C.c_size_t()
        self.check(logsize(prog, C.byref(size)))
        log = C.create_string_buffer(size.value)
        self.check(getlog(prog, log))
        if err:
            raise RuntimeError(log.value.decode())
        self.check(getsize(prog, C.byref(size)))
        ptx = C.create_string_buffer(size.value)
        self.check(getptx(prog, ptx))
        self.check(destroy(C.byref(prog)))

        torch.cuda.current_stream()
        self.module = void()
        self.function = void()
        self.check(
            bind(self.driver, "cuModuleLoadData", [ptr, void])(
                C.byref(self.module), ptx
            )
        )
        self.check(
            bind(self.driver, "cuModuleGetFunction", [ptr, void, C.c_char_p])(
                C.byref(self.function), self.module, b"snake"
            )
        )
        self.launch = bind(
            self.driver,
            "cuLaunchKernel",
            [
                void,
                C.c_uint,
                C.c_uint,
                C.c_uint,
                C.c_uint,
                C.c_uint,
                C.c_uint,
                C.c_uint,
                void,
                ptr,
                void,
            ],
        )

    @staticmethod
    def check(err):
        if err:
            raise RuntimeError(f"CUDA/NVRTC error {err}")

    def __call__(self, x, a):
        assert x.is_cuda and a.is_cuda and x.device == a.device
        assert x.dtype == a.dtype == torch.float32
        assert x.is_contiguous() and a.is_contiguous()
        assert x.ndim == 3 and a.shape == (1, x.shape[1], 1)
        assert not torch.is_grad_enabled() and x.numel() < 2147483647
        y = torch.empty_like(x)
        values = [
            C.c_uint64(x.data_ptr()),
            C.c_uint64(a.data_ptr()),
            C.c_uint64(y.data_ptr()),
            C.c_int(x.numel()),
            C.c_int(x.shape[1]),
            C.c_int(x.shape[2]),
        ]
        args = (C.c_void_p * len(values))(
            *[C.cast(C.byref(v), C.c_void_p) for v in values]
        )
        self.check(
            self.launch(
                self.function,
                (x.numel() + 255) // 256,
                1,
                1,
                256,
                1,
                1,
                0,
                torch.cuda.current_stream().cuda_stream,
                args,
                None,
            )
        )
        return y


def install(model, kernel):
    """Replace AdaINResBlock1.forward with fused Snake activations."""

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = kernel(n1(x, s), a1)
            xt = c1(xt)
            xt = kernel(n2(xt, s), a2)
            xt = c2(xt)
            x = xt + x
        return x

    names = []
    for name, module in model.named_modules():
        if type(module).__name__ == "AdaINResBlock1":
            module.forward = types.MethodType(forward, module)
            names.append(name)
    return names
