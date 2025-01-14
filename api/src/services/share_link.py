import time
from loguru import logger

def start_cloudflared(port: int, max_attempts: int = 3):
    try:
        from flask_cloudflared import _run_cloudflared
    except ImportError:
        logger.info('You should install flask_cloudflared manually')
        raise Exception(
            'flask_cloudflared not installed. Make sure you installed flask_cloudflared==0.0.14')
    
    for _ in range(max_attempts):
        try:
            public_url = _run_cloudflared(port, port + 1)
            logger.info(f'OpenAI-compatible API URL:\n\n{public_url}\n\n{public_url}/v1/audio/speech\n')
            
            return
        except Exception:
            logger.info('starting cloudflared  Failed, retrying in 3 seconds...')
            time.sleep(3)
    logger.error(f'starting cloudflared  Failed , after {max_attempts} attempts')
    # raise Exception('Could not start cloudflared.')
