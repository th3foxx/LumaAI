# utils/mpc_cli.py

import asyncio
import subprocess
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

async def _run_mpc_command_async(args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Асинхронно выполняет команду MPC в отдельном потоке, чтобы не блокировать event loop.
    """
    def sync_run():
        # Эта вложенная функция будет выполняться в executor'е
        try:
            # Увеличиваем таймаут для команд, которые могут быть долгими
            timeout_duration = 30 if args and args[0] in ['listall', 'add', 'update', 'findadd'] else 10
            
            process = subprocess.Popen(
                ['mpc'] + args, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                encoding='utf-8'
            )
            stdout, stderr = process.communicate(timeout=timeout_duration)
            
            # Некоторые команды mpc (например, 'current' или 'playlist', когда плейлист пуст)
            # могут ничего не выводить в stdout и иметь код возврата 1, но это не ошибка.
            is_non_error_empty_stdout = args and args[0] in ['current', 'playlist'] and not stdout.strip() and not stderr.strip()
            
            if process.returncode != 0 and not is_non_error_empty_stdout:
                err_msg = stderr.strip() if stderr else f"MPC command failed with code {process.returncode}"
                logger.warning(f"MPC command {' '.join(args)} error (code {process.returncode}): {err_msg}. Stdout: {stdout.strip() if stdout else 'empty'}")
                return stdout.strip() if stdout else None, err_msg
            
            if stderr and stderr.strip() and process.returncode == 0:
                logger.info(f"MPC command {' '.join(args)} successful with stderr info: {stderr.strip()}")

            return stdout.strip() if stdout else None, None
        
        except FileNotFoundError:
            logger.error("MPC command not found. Is 'mpc' installed and in your system's PATH?")
            return None, "MPC command not found."
        except subprocess.TimeoutExpired:
            logger.error(f"MPC command {' '.join(args)} timed out after {timeout_duration} seconds.")
            return None, "MPC command timed out."
        except Exception as e:
            logger.error(f"Unexpected error running MPC command {' '.join(args)}: {e}", exc_info=True)
            return None, f"Unexpected MPC error: {str(e)}"

    loop = asyncio.get_running_loop()
    # Запускаем синхронную функцию в отдельном потоке
    return await loop.run_in_executor(None, sync_run)


# --- Публичные функции-обертки ---

async def mpc_update_library() -> bool:
    """Запускает обновление библиотеки MPD."""
    logger.info("Attempting to trigger MPD library update (mpc update)...")
    _, stderr = await _run_mpc_command_async(['update'])
    if stderr:
        logger.error(f"Error triggering MPD library update: {stderr}")
        return False
    logger.info("MPD library update command sent successfully.")
    return True

async def mpc_get_current_track_details() -> Optional[dict]:
    """Получает детали текущего трека из MPD."""
    stdout, err = await _run_mpc_command_async(['current', '-f', '%file%\\t%artist%\\t%title%\\t%album%'])
    if err or not stdout:
        status_out, _ = await _run_mpc_command_async(['status'])
        if status_out and ("volume:" in status_out and not "[playing]" in status_out and not "[paused]" in status_out):
            logger.debug("MPD: Nothing seems to be playing (checked via status).")
            return None
        if err:
            logger.warning(f"Could not get current MPD track details: {err}")
        return None

    parts = stdout.split('\t', 3)
    file_path = parts[0]
    artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
    title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else None
    album = parts[3] if len(parts) > 3 and parts[3] != '(null)' else None
    
    if not title and file_path and not ("http://" in file_path or "https://" in file_path):
        try:
            title = file_path.split('/')[-1].rsplit('.', 1)[0]
        except:
            title = file_path

    return {"file": file_path, "artist": artist, "title": title, "album": album}

async def mpc_get_status() -> Optional[str]:
    """Получает статус плеера MPD."""
    stdout, _ = await _run_mpc_command_async(['status'])
    return stdout

async def mpc_get_playlist() -> Optional[str]:
    """Получает содержимое текущего плейлиста MPD."""
    stdout, _ = await _run_mpc_command_async(['playlist'])
    return stdout

async def mpc_add_to_playlist(uri: str) -> Tuple[bool, Optional[str]]:
    """Добавляет URI в плейлист MPD."""
    _, stderr = await _run_mpc_command_async(['add', uri])
    return not stderr, stderr

async def mpc_find_and_add(filters: List[str]) -> Tuple[bool, Optional[str]]:
    """Ищет треки по фильтрам и добавляет их в плейлист."""
    _, stderr = await _run_mpc_command_async(['findadd'] + filters)
    return not stderr, stderr

async def mpc_load_playlist(playlist_name: str) -> Tuple[bool, Optional[str]]:
    """Загружает сохраненный плейлист MPD."""
    _, stderr = await _run_mpc_command_async(['load', playlist_name])
    return not stderr, stderr

async def mpc_list_all_tracks() -> Optional[List[str]]:
    """Возвращает список всех треков в библиотеке MPD."""
    stdout, err = await _run_mpc_command_async(['listall'])
    if err or not stdout:
        logger.error(f"Failed to list all local tracks for random play: {err}")
        return None
    return [track for track in stdout.splitlines() if track.strip()]

async def mpc_simple_command(command: str, *args: str) -> Tuple[bool, Optional[str]]:
    """Выполняет простую команду MPD (play, pause, stop, clear, shuffle, next, prev, volume)."""
    _, stderr = await _run_mpc_command_async([command, *args])
    return not stderr, stderr