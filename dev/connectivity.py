# --- START OF MODIFIED FILE connectivity.py ---
import asyncio
import socket
import time
import logging

logger = logging.getLogger(__name__)

# --- Connectivity Cache & State ---
_last_check_time: float = 0.0
_last_check_result: bool | None = None # Legacy cache, can be phased out or kept as secondary
_cache_duration: float = 60.0 # Legacy cache duration

_check_host: str = "8.8.8.8"
_check_port: int = 53
_check_timeout: float = 1.5 # Timeout for the socket connection attempt
_monitoring_interval: float = 15.0 # How often the background task checks connectivity (seconds)

# Global state for background monitoring
_internet_is_currently_available: bool | None = None # Updated by background task
_connectivity_monitoring_task: asyncio.Task | None = None
_stop_monitoring_event = asyncio.Event() # Used to signal the monitoring task to stop

def set_connectivity_check_params(
    host: str = "8.8.8.8",
    port: int = 53,
    timeout: float = 1.5,
    monitoring_interval: float = 15.0, # Added monitoring interval
    cache_duration: float = 60.0 # Kept for compatibility if direct checks are still used
):
    """Allows overriding default connectivity check parameters."""
    global _check_host, _check_port, _check_timeout, _monitoring_interval, _cache_duration
    _check_host = host
    _check_port = port
    _check_timeout = timeout
    _monitoring_interval = monitoring_interval
    _cache_duration = cache_duration

    # Invalidate legacy cache on parameter change
    global _last_check_time, _last_check_result
    _last_check_time = 0.0
    _last_check_result = None
    logger.info(
        f"Connectivity check params updated: Host={host}, Port={port}, Timeout={timeout}s, "
        f"Monitoring Interval={_monitoring_interval}s, Cache Duration={_cache_duration}s"
    )


async def _perform_check() -> bool:
    """Performs the actual network check."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(_check_host, _check_port),
            timeout=_check_timeout
        )
        writer.close()
        await writer.wait_closed()
        logger.debug(f"Connectivity check to {_check_host}:{_check_port} successful.")
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Connectivity check to {_check_host}:{_check_port} timed out after {_check_timeout}s.")
        return False
    except (socket.gaierror, ConnectionRefusedError, OSError) as e:
        logger.warning(f"Connectivity check to {_check_host}:{_check_port} failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during connectivity check: {e}", exc_info=True)
        return False

async def _monitor_connectivity():
    """Background task to periodically check internet connectivity."""
    global _internet_is_currently_available
    logger.info("Starting background internet connectivity monitor...")
    while not _stop_monitoring_event.is_set():
        try:
            current_status = await _perform_check()
            if _internet_is_currently_available != current_status:
                logger.info(f"Internet connectivity status changed to: {current_status}")
                _internet_is_currently_available = current_status
            else:
                logger.debug(f"Internet connectivity status remains: {current_status}")

            # Wait for the next interval or until stop event is set
            await asyncio.wait_for(
                _stop_monitoring_event.wait(),
                timeout=_monitoring_interval
            )
        except asyncio.TimeoutError:
            # This is expected, means _monitoring_interval passed
            continue
        except Exception as e:
            logger.error(f"Error in connectivity monitoring loop: {e}", exc_info=True)
            # Avoid busy-looping on persistent errors, wait a bit before retrying
            await asyncio.sleep(min(_monitoring_interval, 30)) # Wait but not too long
    logger.info("Background internet connectivity monitor stopped.")

async def start_connectivity_monitoring():
    """Starts the background connectivity monitoring task."""
    global _connectivity_monitoring_task, _internet_is_currently_available
    if _connectivity_monitoring_task is None or _connectivity_monitoring_task.done():
        # Perform an initial check immediately
        _internet_is_currently_available = await _perform_check()
        logger.info(f"Initial internet connectivity status: {_internet_is_currently_available}")

        _stop_monitoring_event.clear()
        _connectivity_monitoring_task = asyncio.create_task(_monitor_connectivity())
        logger.info("Connectivity monitoring task created and started.")
    else:
        logger.info("Connectivity monitoring task is already running.")

async def stop_connectivity_monitoring():
    """Stops the background connectivity monitoring task."""
    global _connectivity_monitoring_task
    if _connectivity_monitoring_task and not _connectivity_monitoring_task.done():
        logger.info("Stopping connectivity monitoring task...")
        _stop_monitoring_event.set()
        try:
            await asyncio.wait_for(_connectivity_monitoring_task, timeout=_monitoring_interval + 5)
            logger.info("Connectivity monitoring task successfully stopped.")
        except asyncio.TimeoutError:
            logger.warning("Connectivity monitoring task did not stop in time, cancelling.")
            _connectivity_monitoring_task.cancel()
        except Exception as e:
            logger.error(f"Error stopping connectivity monitoring task: {e}", exc_info=True)
        _connectivity_monitoring_task = None
    else:
        logger.info("Connectivity monitoring task not running or already stopped.")

async def is_internet_available(force_check: bool = False) -> bool:
    """
    Checks for internet connectivity.
    Uses the state from the background monitor if available,
    otherwise falls back to a direct check with caching.

    Args:
        force_check: If True, performs a new direct check, bypassing monitor and cache.

    Returns:
        True if internet connection is likely available, False otherwise.
    """
    global _last_check_time, _last_check_result, _internet_is_currently_available

    if force_check:
        logger.debug("Forcing new connectivity check...")
        result = await _perform_check()
        _last_check_time = time.monotonic() # Update legacy cache time as well
        _last_check_result = result
        # Also update the global monitor state if we force a check,
        # as it's the most up-to-date info we have.
        if _internet_is_currently_available != result:
             logger.info(f"Forced check updated connectivity status to: {result}")
        _internet_is_currently_available = result
        return result

    # Prefer the background monitor's state if it has been initialized
    if _internet_is_currently_available is not None:
        logger.debug(f"Returning connectivity status from background monitor: {_internet_is_currently_available}")
        return _internet_is_currently_available

    # Fallback to legacy cache if monitor hasn't run yet (e.g., very early startup)
    now = time.monotonic()
    if _last_check_result is not None and (now - _last_check_time) < _cache_duration:
        logger.debug(f"Returning cached connectivity status (monitor not ready): {_last_check_result}")
        return _last_check_result

    # If monitor not ready and cache stale/empty, perform a direct check
    logger.debug("Performing new connectivity check (monitor not ready, cache stale)...")
    result = await _perform_check()
    _last_check_time = now
    _last_check_result = result
    # Potentially initialize the monitored state here too if it's still None
    if _internet_is_currently_available is None:
        _internet_is_currently_available = result
    logger.info(f"Direct connectivity check result: {result} (Cached for {_cache_duration}s)")
    return result

# --- END OF MODIFIED FILE connectivity.py ---