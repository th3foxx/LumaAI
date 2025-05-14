import asyncio
import socket
import time
import logging

logger = logging.getLogger(__name__)

# --- Connectivity Cache ---
_last_check_time: float = 0.0
_last_check_result: bool | None = None
_cache_duration: float = 60.0 # Cache result for 60 seconds
_check_host: str = "8.8.8.8" # Google's public DNS
_check_port: int = 53
_check_timeout: float = 1.5 # Seconds

def set_connectivity_check_params(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.5, cache_duration: float = 60.0):
    """Allows overriding default connectivity check parameters."""
    global _check_host, _check_port, _check_timeout, _cache_duration
    _check_host = host
    _check_port = port
    _check_timeout = timeout
    _cache_duration = cache_duration
    # Invalidate cache on parameter change
    global _last_check_time, _last_check_result
    _last_check_time = 0.0
    _last_check_result = None
    logger.info(f"Connectivity check params updated: Host={host}, Port={port}, Timeout={timeout}s, Cache={cache_duration}s")


async def _perform_check() -> bool:
    """Performs the actual network check."""
    try:
        # Use asyncio's loop to run the blocking socket operation in a thread
        loop = asyncio.get_running_loop()
        # Create a socket connection in a non-blocking way via executor
        transport, protocol = await loop.create_connection(
            lambda: asyncio.Protocol(), # Simple protocol, we just need the connection
            _check_host,
            _check_port
        )
        # If connection succeeds, close it immediately and return True
        transport.close()
        logger.debug(f"Connectivity check to {_check_host}:{_check_port} successful.")
        return True
    except (socket.gaierror, ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
        # Handle common errors indicating no connection or resolution failure
        logger.warning(f"Connectivity check to {_check_host}:{_check_port} failed: {e}")
        return False
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during connectivity check: {e}", exc_info=True)
        return False # Assume no connection on unexpected errors

async def is_internet_available(force_check: bool = False) -> bool:
    """
    Checks for internet connectivity with caching.

    Args:
        force_check: If True, ignores the cache and performs a new check.

    Returns:
        True if internet connection is likely available, False otherwise.
    """
    global _last_check_time, _last_check_result

    now = time.monotonic()
    if not force_check and _last_check_result is not None and (now - _last_check_time) < _cache_duration:
        logger.debug(f"Returning cached connectivity status: {_last_check_result}")
        return _last_check_result

    logger.debug("Performing new connectivity check...")
    result = await _perform_check()

    _last_check_time = now
    _last_check_result = result
    logger.info(f"Connectivity status updated: {result} (Cached for {_cache_duration}s)")
    return result