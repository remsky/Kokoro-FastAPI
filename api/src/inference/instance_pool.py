"""GPU instance pool and request queue management."""

import asyncio
from typing import Optional, Dict, List, Any
import torch
from loguru import logger

from ..core.config import settings
from .model_manager import ModelManager, get_manager


class GPUInstance:
    """Represents a model instance running on a specific GPU."""
    
    def __init__(self, device_id: int, instance_id: int):
        self.device_id = device_id
        self.instance_id = instance_id  # Instance ID for the same GPU
        self.manager: Optional[ModelManager] = None
        self.is_busy: bool = False
        self.current_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the model instance on the specified GPU."""
        try:
            # Set CUDA device
            torch.cuda.set_device(self.device_id)
            # Create a new model manager instance for this GPU instance
            self.manager = await get_manager()
            # Initialize with warmup
            await self.manager.initialize_with_warmup(None)
            logger.info(f"Initialized model instance {self.instance_id} on GPU {self.device_id}")
        except Exception as e:
            logger.error(f"Failed to initialize instance {self.instance_id} on GPU {self.device_id}: {e}")
            raise
    
    def __del__(self):
        """Cleanup when instance is destroyed."""
        if self.manager:
            self.manager.unload_all()
            self.manager = None


class InstancePool:
    """Manages multiple GPU instances and request queue."""
    
    _instance = None
    
    def __init__(self):
        self.instances: List[GPUInstance] = []
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.request_queue_size)
        self.current_instance_idx = 0
    
    @classmethod
    async def get_instance(cls) -> 'InstancePool':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize GPU instances."""
        # Create multiple instances on the same GPU
        for i in range(settings.instances_per_gpu):
            instance = GPUInstance(settings.gpu_device, i)
            try:
                await instance.initialize()
                self.instances.append(instance)
                logger.info(f"Successfully initialized instance {i} on GPU {settings.gpu_device}")
            except Exception as e:
                logger.error(f"Failed to initialize instance {i}: {e}")
                # If we failed to initialize any instance, cleanup and raise
                if not self.instances:
                    raise RuntimeError("Failed to initialize any GPU instances")
                break
            
        if not self.instances:
            raise RuntimeError("No GPU instances initialized")
        
        logger.info(f"Successfully initialized {len(self.instances)} instances on GPU {settings.gpu_device}")
        
        # Start request processor
        asyncio.create_task(self._process_queue())
    
    def get_next_available_instance(self) -> Optional[GPUInstance]:
        """Get next available GPU instance using round-robin."""
        start_idx = self.current_instance_idx
        
        # Try to find an available instance
        for _ in range(len(self.instances)):
            instance = self.instances[self.current_instance_idx]
            self.current_instance_idx = (self.current_instance_idx + 1) % len(self.instances)
            
            if not instance.is_busy:
                return instance
                
            # If we're back at start, no instance is available
            if self.current_instance_idx == start_idx:
                return None
        
        return None
    
    async def _process_queue(self) -> None:
        """Process requests from queue."""
        while True:
            try:
                # Get request from queue
                request = await self.request_queue.get()
                text, voice_info = request["text"], request["voice_info"]
                speed = request.get("speed", 1.0)  # Get speed with default 1.0
                future = request["future"]
                
                # Get available instance
                instance = self.get_next_available_instance()
                if instance is None:
                    # No instance available, put back in queue
                    await self.request_queue.put(request)
                    await asyncio.sleep(0.1)
                    continue
                
                # Mark instance as busy
                instance.is_busy = True
                logger.debug(f"Processing request on instance {instance.instance_id}")
                
                try:
                    # Process request
                    result = []
                    async for chunk in instance.manager.generate(text, voice_info, speed=speed):
                        result.append(chunk)
                    future.set_result(result)
                except Exception as e:
                    logger.error(f"Error in instance {instance.instance_id}: {e}")
                    future.set_exception(e)
                finally:
                    # Mark instance as available
                    instance.is_busy = False
                    self.request_queue.task_done()
                    logger.debug(f"Instance {instance.instance_id} is now available")
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await asyncio.sleep(1)
    
    async def process_request(self, text: str, voice_info: tuple, speed: float = 1.0) -> List[Any]:
        """Submit request to queue and wait for result."""
        # Create future to get result
        future = asyncio.Future()
        
        # Create request
        request = {
            "text": text,
            "voice_info": voice_info,
            "speed": speed,
            "future": future
        }
        
        try:
            # Put request in queue with timeout
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=settings.request_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Request queue is full")
        
        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(
                future,
                timeout=settings.request_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("Request processing timed out") 