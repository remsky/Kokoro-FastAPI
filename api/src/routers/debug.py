import threading
import time
from datetime import datetime

import psutil
import torch
from fastapi import APIRouter

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

router = APIRouter(tags=["debug"])


@router.get("/debug/threads")
async def get_thread_info():
    process = psutil.Process()
    current_threads = threading.enumerate()

    # Get per-thread CPU times
    thread_details = []
    for thread in current_threads:
        thread_info = {
            "name": thread.name,
            "id": thread.ident,
            "alive": thread.is_alive(),
            "daemon": thread.daemon,
        }
        thread_details.append(thread_info)

    return {
        "total_threads": process.num_threads(),
        "active_threads": len(current_threads),
        "thread_names": [t.name for t in current_threads],
        "thread_details": thread_details,
        "memory_mb": process.memory_info().rss / 1024 / 1024,
    }


@router.get("/debug/storage")
async def get_storage_info():
    # Get disk partitions
    partitions = psutil.disk_partitions()
    storage_info = []

    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            storage_info.append(
                {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent_used": usage.percent,
                }
            )
        except PermissionError:
            continue

    return {"storage_info": storage_info}


@router.get("/debug/system")
async def get_system_info():
    process = psutil.Process()

    # CPU Info
    cpu_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "per_cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
        "load_avg": psutil.getloadavg(),
    }

    # Memory Info
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    memory_info = {
        "virtual": {
            "total_gb": virtual_memory.total / (1024**3),
            "available_gb": virtual_memory.available / (1024**3),
            "used_gb": virtual_memory.used / (1024**3),
            "percent": virtual_memory.percent,
        },
        "swap": {
            "total_gb": swap_memory.total / (1024**3),
            "used_gb": swap_memory.used / (1024**3),
            "free_gb": swap_memory.free / (1024**3),
            "percent": swap_memory.percent,
        },
    }

    # Process Info
    process_info = {
        "pid": process.pid,
        "status": process.status(),
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent(),
    }

    # Network Info
    network_info = {
        "connections": len(process.net_connections()),
        "network_io": psutil.net_io_counters()._asdict(),
    }

    # GPU Info if available
    gpu_info = None
    if torch.backends.mps.is_available():
        gpu_info = {
            "type": "MPS",
            "available": True,
            "device": "Apple Silicon",
            "backend": "Metal",
        }
    elif GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                        "percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    },
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ]
        except Exception:
            gpu_info = "GPU information unavailable"

    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "process": process_info,
        "network": network_info,
        "gpu": gpu_info,
    }


@router.get("/debug/session_pools")
async def get_session_pool_info():
    """Get information about ONNX session pools."""
    from ..inference.model_manager import get_manager

    manager = await get_manager()
    pools = manager._session_pools
    current_time = time.time()

    pool_info = {}

    # Get CPU pool info
    if "onnx_cpu" in pools:
        cpu_pool = pools["onnx_cpu"]
        pool_info["cpu"] = {
            "active_sessions": len(cpu_pool._sessions),
            "max_sessions": cpu_pool._max_size,
            "sessions": [
                {"model": path, "age_seconds": current_time - info.last_used}
                for path, info in cpu_pool._sessions.items()
            ],
        }

    # Get GPU pool info
    if "onnx_gpu" in pools:
        gpu_pool = pools["onnx_gpu"]
        pool_info["gpu"] = {
            "active_sessions": len(gpu_pool._sessions),
            "max_streams": gpu_pool._max_size,
            "available_streams": len(gpu_pool._available_streams),
            "sessions": [
                {
                    "model": path,
                    "age_seconds": current_time - info.last_used,
                    "stream_id": info.stream_id,
                }
                for path, info in gpu_pool._sessions.items()
            ],
        }

        # Add GPU memory info if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assume first GPU
                    pool_info["gpu"]["memory"] = {
                        "total_mb": gpu.memoryTotal,
                        "used_mb": gpu.memoryUsed,
                        "free_mb": gpu.memoryFree,
                        "percent_used": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    }
            except Exception:
                pass

    return pool_info
