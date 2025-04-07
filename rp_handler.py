import runpod
import json
import asyncio
from loguru import logger
import time

# 导入路径
from api.src.services.tts_service import TTSService
from api.src.services.audio import AudioService

# 全局变量
tts_service = None
audio_service = None
_init_lock = None  # 添加初始化锁，确保并发安全

# 加载OpenAI映射
def load_openai_mappings():
    """Load OpenAI voice and model mappings from JSON"""
    try:
        mapping_path = "api/src/core/openai_mappings.json"
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI mappings: {e}")
        return {"models": {}, "voices": {}}

# 初始化全局变量
openai_mappings = load_openai_mappings()

# 获取TTS服务（带并发锁保护）
async def get_tts_service():
    """获取全局TTS服务实例，确保只初始化一次"""
    global tts_service, audio_service, _init_lock
    
    # 创建锁（如果需要）
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    
    # 初始化服务（如果需要）
    if tts_service is None:
        async with _init_lock:  # 使用锁防止并发初始化
            # 双重检查模式
            if tts_service is None:
                try:
                    logger.info("Creating TTS service...")
                    tts_service = await TTSService.create()
                    audio_service = AudioService()
                    
                    # 验证初始化成功
                    if tts_service.model_manager is None or tts_service._voice_manager is None:
                        raise RuntimeError("TTS service backend not initialized")
                        
                    # 通过列出声音测试服务是否正常工作
                    await tts_service.list_voices()
                    logger.info("TTS service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize TTS service: {e}", exc_info=True)
                    tts_service = None  # 重置为None以便下次尝试
                    audio_service = None
                    raise RuntimeError(f"Failed to initialize TTS service: {str(e)}")
    
    return tts_service

async def process_voices(voice_input, available_voices):
    """处理声音输入，支持字符串和列表格式"""
    # 如果是字符串输入
    if isinstance(voice_input, str):
        # 检查是否是OpenAI声音名称
        mapped_voice = openai_mappings["voices"].get(voice_input)
        if mapped_voice:
            voice_input = mapped_voice
            
        # 分割声音，保留括号内的权重
        voices = []
        for part in voice_input.split("+"):
            part = part.strip()
            if not part:
                continue
                
            # 提取不带权重的声音名称
            voice_name = part.split("(")[0].strip()
            
            # 验证声音是否存在
            if voice_name not in available_voices:
                raise ValueError(
                    f"Voice '{voice_name}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )
            voices.append(part)
    else:
        # 处理列表输入
        voices = []
        for v in voice_input:
            mapped = openai_mappings["voices"].get(v, v)
            voice_name = mapped.split("(")[0].strip()
            
            # 验证声音是否存在
            if voice_name not in available_voices:
                raise ValueError(
                    f"Voice '{voice_name}' not found. Available voices: {', '.join(sorted(available_voices))}"
                )
            voices.append(mapped)

    if not voices:
        raise ValueError("No voices provided")

    # 组合多个声音
    return "+".join(voices)

async def stream_audio_generation(text, voice_name, speed, response_format, lang_code):
    """流式生成音频的包装函数，与openai_compatible.py逻辑保持一致"""
    try:
        logger.info(f"Starting audio generation with lang_code: {lang_code}")
        # 获取TTS服务
        service = await get_tts_service()
        
        # 使用流式API生成音频
        async for chunk in service.generate_audio_stream(
            text=text,
            voice=voice_name,
            speed=speed,
            output_format=response_format,
            lang_code=lang_code,
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        raise

async def handler(event):
    """RunPod异步处理程序，用于文本到语音转换"""
    try:
        input_data = event.get("input", {})
        
        # 提取参数 - 使用schemas.py中定义的默认值
        text = input_data.get("input")
        model = input_data.get("model", "kokoro")  # 默认为kokoro
        voice = input_data.get("voice", "af_alloy")  # 默认为af_alloy
        response_format = input_data.get("response_format", "mp3")  # 默认为mp3
        
        # 验证speed在有效范围内 (0.25-4.0)
        speed = float(input_data.get("speed", 1.0))  # 默认为1.0
        if speed < 0.25 or speed > 4.0:
            yield {
                "status": "error", 
                "error": "invalid_speed",
                "message": "Speed must be between 0.25 and 4.0"
            }
            return
        
        lang_code = input_data.get("lang_code")  # 默认为None
        
        # 验证必需参数
        if not text:
            yield {"status": "error", "error": "Text input is required"}
            return
        
        # 验证模型
        if model not in openai_mappings["models"]:
            yield {
                "status": "error",
                "error": "invalid_model",
                "message": f"Unsupported model: {model}"
            }
            return
        
        # 验证格式并获取content_type
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if response_format not in valid_formats:
            yield {
                "status": "error",
                "error": "invalid_format",
                "message": f"Unsupported format: {response_format}. Supported formats: {', '.join(valid_formats)}"
            }
            return
        
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        content_type = content_type_map.get(response_format)
        
        try:
            # 初始状态更新
            yield {"status": "initializing", "progress": 10, "message": "Initializing services..."}
            
            # 获取TTS服务（会自动处理初始化）
            try:
                service = await get_tts_service()
            except RuntimeError as e:
                yield {"status": "error", "error": str(e)}
                return
                
            yield {"status": "processing", "progress": 25, "message": "Processing voice configuration..."}
            
            # 获取可用声音列表
            available_voices = await service.list_voices()
            
            # 处理声音输入
            voice_name = await process_voices(voice, available_voices)
            
            # 确定语言代码
            effective_lang_code = lang_code
            if not effective_lang_code:
                if isinstance(voice, list) and voice:
                    effective_lang_code = voice[0]
                elif isinstance(voice, str):
                    effective_lang_code = voice
            
            # 使用流式生成音频
            yield {"status": "generating", "progress": 40, "message": "Generating audio content..."}
            
            # 记录开始时间用于性能测量
            start_time = time.time()
            
            # 返回音频元数据 
            yield {
                "status": "stream_start",
                "content_type": content_type,
                "format": response_format,
                "sample_rate": 24000
            }
            
            chunk_count = 0
            total_size = 0
            
            # 真正的流式处理 - 直接返回每个音频块
            async for audio_chunk in stream_audio_generation(
                text=text,
                voice_name=voice_name,
                speed=speed,
                response_format=response_format,
                lang_code=effective_lang_code
            ):
                if audio_chunk is not None:
                    chunk_count += 1
                    chunk_size = len(audio_chunk)
                    total_size += chunk_size
                    
                    # 直接返回二进制音频数据
                    yield {
                        "status": "audio_chunk",
                        "chunk_index": chunk_count,
                        "audio_data": audio_chunk,  # RunPod会自动处理二进制数据
                        "chunk_size": chunk_size
                    }
                    
                    # 仍然提供进度更新（作为单独的消息）
                    progress = min(40 + int(chunk_count * 50 / max(len(text) // 100, 1)), 90)
                    yield {
                        "status": "progress", 
                        "progress": progress, 
                        "message": f"Generated chunk {chunk_count}"
                    }
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 返回完成状态
            yield {
                "status": "complete",
                "progress": 100,
                "format": response_format,
                "content_type": content_type,
                "processing_time": processing_time,
                "total_chunks": chunk_count,
                "total_size": total_size
            }
                
        except ValueError as e:
            # 处理验证错误
            logger.warning(f"Validation error: {str(e)}")
            yield {"status": "error", "error": f"Validation error: {str(e)}"}
        except RuntimeError as e:
            # 处理运行时错误
            logger.error(f"Runtime error: {str(e)}")
            yield {"status": "error", "error": f"Runtime error: {str(e)}"}
        except Exception as e:
            # 处理其他错误
            logger.error(f"Error in audio processing: {e}", exc_info=True)
            yield {"status": "error", "error": f"Audio processing error: {str(e)}"}
            
    except Exception as e:
        # 处理顶层错误
        logger.error(f"Error in TTS processing: {e}", exc_info=True)
        yield {"status": "error", "error": f"TTS processing error: {str(e)}"}

if __name__ == '__main__':
    # 启动RunPod服务
    runpod.serverless.start({
        'handler': handler,
        'return_aggregate_stream': True  # 启用聚合流结果
    })