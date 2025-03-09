import asyncio
import time
from datetime import datetime
import aiohttp
import json
from loguru import logger
import numpy as np

# Test configuration
BASE_URL = "http://localhost:50888"
API_KEY = "sk-kokoro-f7a9b2c8e6d4g3h1j5k0"
CONCURRENT_REQUESTS = 3  # Number of concurrent requests
TOTAL_REQUESTS = 100     # Total number of requests to make
TEST_DURATION = 60       # Test duration in seconds
CHUNK_SIZE = 8192       # Increased chunk size for better performance

# Test payload
TEST_PAYLOAD = {
    "model": "kokoro",
    "input": "This is a performance test text.",  # Shorter test text
    "voice": "af_heart",
    "response_format": "mp3",
    "download_format": "mp3",
    "speed": 1,
    "stream": True,
    "return_download_link": False,
    "lang_code": "a"
}

class PerformanceMetrics:
    def __init__(self):
        self.request_times = []
        self.audio_sizes = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None
        self.current_requests = 0
        self.max_concurrent = 0

    def add_request(self, duration, audio_size=0, success=True):
        self.request_times.append(duration)
        if audio_size > 0:
            self.audio_sizes.append(audio_size)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def update_concurrent(self, delta):
        self.current_requests += delta
        self.max_concurrent = max(self.max_concurrent, self.current_requests)

    def calculate_metrics(self):
        test_duration = (self.end_time - self.start_time).total_seconds()
        qps = self.success_count / test_duration if test_duration > 0 else 0
        avg_latency = np.mean(self.request_times) if self.request_times else 0
        p95_latency = np.percentile(self.request_times, 95) if self.request_times else 0
        p99_latency = np.percentile(self.request_times, 99) if self.request_times else 0
        
        total_audio_mb = sum(self.audio_sizes) / (1024 * 1024)
        audio_throughput = total_audio_mb / test_duration if test_duration > 0 else 0

        return {
            "qps": qps,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100,
            "audio_throughput_mbps": audio_throughput,
            "total_requests": self.success_count + self.error_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "test_duration": test_duration,
            "max_concurrent": self.max_concurrent
        }

async def make_request(session, metrics, semaphore, request_id):
    try:
        async with semaphore:
            metrics.update_concurrent(1)
            start_time = time.time()
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Accept": "audio/mpeg"
            }
            
            try:
                async with session.post(
                    f"{BASE_URL}/v1/audio/speech",
                    json=TEST_PAYLOAD,
                    headers=headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        total_size = 0
                        audio_data = bytearray()
                        
                        try:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                if chunk:  # Only process non-empty chunks
                                    audio_data.extend(chunk)
                                    total_size += len(chunk)
                            
                            duration = time.time() - start_time
                            metrics.add_request(duration, total_size, True)
                            logger.debug(f"Request {request_id} completed successfully: {total_size} bytes in {duration:.2f}s")
                            return True
                            
                        except Exception as chunk_error:
                            logger.error(f"Chunk processing error in request {request_id}: {str(chunk_error)}")
                            duration = time.time() - start_time
                            metrics.add_request(duration, success=False)
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Request {request_id} failed with status {response.status}: {error_text}")
                        duration = time.time() - start_time
                        metrics.add_request(duration, success=False)
                        return False
                        
            except asyncio.TimeoutError:
                logger.error(f"Request {request_id} timed out")
                duration = time.time() - start_time
                metrics.add_request(duration, success=False)
                return False
                
            except Exception as e:
                logger.error(f"Request {request_id} failed with error: {str(e)}")
                duration = time.time() - start_time
                metrics.add_request(duration, success=False)
                return False
                
    finally:
        metrics.update_concurrent(-1)

async def run_load_test():
    metrics = PerformanceMetrics()
    metrics.start_time = datetime.now()
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(TOTAL_REQUESTS):
            task = asyncio.create_task(make_request(session, metrics, semaphore, i+1))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
    
    metrics.end_time = datetime.now()
    return metrics

def print_results(metrics_data):
    logger.info("\n=== Performance Test Results ===")
    logger.info(f"Total Requests: {metrics_data['total_requests']}")
    logger.info(f"Successful Requests: {metrics_data['successful_requests']}")
    logger.info(f"Failed Requests: {metrics_data['failed_requests']}")
    logger.info(f"Success Rate: {metrics_data['success_rate']:.2f}%")
    logger.info(f"Test Duration: {metrics_data['test_duration']:.2f} seconds")
    logger.info(f"QPS: {metrics_data['qps']:.2f}")
    logger.info(f"Average Latency: {metrics_data['avg_latency']*1000:.2f} ms")
    logger.info(f"P95 Latency: {metrics_data['p95_latency']*1000:.2f} ms")
    logger.info(f"P99 Latency: {metrics_data['p99_latency']*1000:.2f} ms")
    logger.info(f"Audio Throughput: {metrics_data['audio_throughput_mbps']:.2f} MB/s")
    logger.info(f"Max Concurrent Requests: {metrics_data['max_concurrent']}")

if __name__ == "__main__":
    logger.info("Starting performance test...")
    metrics = asyncio.run(run_load_test())
    print_results(metrics.calculate_metrics()) 