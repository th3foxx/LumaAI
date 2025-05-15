import asyncio
import subprocess
import logging
from typing import Optional, List, Tuple, Dict, Any
import re
import json

from langchain_core.tools import tool
from settings import settings # Для DB_PATH и других настроек, если понадобятся

# Импортируем функции для работы с БД из нового модуля
from utils.music_db import (
    add_song_to_liked_db,
    remove_song_from_liked_db,
    get_all_liked_songs_from_db,
    is_song_liked_in_db # Переименовал для ясности, что это DB операция
)

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения информации о текущем треке
_CURRENTLY_PLAYING_INFO: Optional[Dict[str, Any]] = None

# --- Вспомогательные функции _run_mpc_command, _search_youtube_and_get_info, _update_current_playing_info ---
# остаются такими же, как в предыдущем вашем полном примере tools/music_control.py
# ... (скопируйте их сюда) ...
def _run_mpc_command(args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        process = subprocess.Popen(['mpc'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=10)
        is_non_error_empty_stdout = args[0] in ['current', 'playlist'] and not stdout.strip() and not stderr.strip()
        if process.returncode != 0 and not is_non_error_empty_stdout:
            if stderr.strip(): logger.warning(f"MPC command {' '.join(args)} stderr (code {process.returncode}): {stderr.strip()}")
            else: logger.warning(f"MPC command {' '.join(args)} returned code {process.returncode} with empty stderr. Stdout: {stdout.strip()}")
            return stdout.strip() if stdout else None, stderr.strip() if stderr else "MPC command failed"
        return stdout.strip() if stdout else None, stderr.strip() if stderr and stderr.strip() else None
    except FileNotFoundError: logger.error("MPC command not found."); return None, "MPC command not found."
    except subprocess.TimeoutExpired: logger.error(f"MPC command {' '.join(args)} timed out."); return None, "MPC command timed out."
    except Exception as e: logger.error(f"Error running MPC command {' '.join(args)}: {e}", exc_info=True); return None, f"Unexpected MPC error: {str(e)}"

async def _search_youtube_and_get_info(query: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Searching YouTube with yt-dlp for: '{query}'")
    try:
        process = await asyncio.create_subprocess_exec(
            'yt-dlp',
            '--skip-download',
            '--dump-single-json',
            '--default-search', 'ytsearch1:', 
            '--ignore-errors',
            '--no-warnings',
            '--format', 'bestaudio/best', # yt-dlp попытается выбрать лучший аудиопоток
            query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)

        if process.returncode != 0:
            stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()
            logger.error(f"yt-dlp search failed for '{query}' (code {process.returncode}). Stderr: {stderr_str}"); return None
        
        if not stdout_bytes:
            logger.warning(f"yt-dlp returned no stdout for '{query}'. Stderr: {stderr_bytes.decode('utf-8', errors='ignore').strip()}"); return None

        result_info = json.loads(stdout_bytes.decode('utf-8', errors='ignore'))
        logger.debug(f"yt-dlp JSON output for '{query}': {json.dumps(result_info, indent=2, ensure_ascii=False)}")

        video_data_to_process = None

        if result_info.get("_type") == "playlist":
            logger.debug(f"yt-dlp returned a playlist for query '{query}'. Processing first entry.")
            if result_info.get("entries") and len(result_info["entries"]) > 0:
                video_data_to_process = result_info["entries"][0] # Берем первое видео из плейлиста/канала
            else:
                logger.warning(f"Playlist/channel result for '{query}' has no entries.")
                return None
        else: # Предполагаем, что это информация об одном видео
            video_data_to_process = result_info

        if not video_data_to_process:
            logger.warning(f"No video data to process from yt-dlp output for '{query}'.")
            return None

        title = video_data_to_process.get('title', query)
        # yt-dlp с --format bestaudio/best должен сам выбрать лучший URL и поместить его в 'url'
        # для ОБЪЕКТА ВИДЕО (не плейлиста).
        audio_url = video_data_to_process.get('url') 

        # Если 'url' нет на верхнем уровне video_data_to_process, или если мы хотим быть уверены,
        # что это аудио, можно дополнительно проверить 'formats'.
        # Но с '--format bestaudio/best' это часто избыточно, yt-dlp уже должен был выбрать.
        if not audio_url and 'formats' in video_data_to_process:
            logger.debug(f"No top-level 'url' in video data for '{title}', checking formats list...")
            # (Ваша существующая логика выбора из 'formats' может остаться здесь как fallback)
            audio_formats_available = []
            for f_info in video_data_to_process['formats']:
                if f_info.get('url') and f_info.get('acodec') and f_info.get('acodec') != 'none':
                    audio_formats_available.append(f_info)
            
            only_audio_formats = [f for f in audio_formats_available if f.get('vcodec') == 'none']
            target_formats_list = only_audio_formats if only_audio_formats else audio_formats_available

            if target_formats_list:
                preferred_exts = ('m4a', 'opus', 'webm', 'mp3')
                for ext in preferred_exts:
                    for f_info in target_formats_list:
                        if f_info.get('ext') == ext: audio_url = f_info['url']; break
                    if audio_url: break
                if not audio_url: audio_url = target_formats_list[0]['url']
        
        if audio_url:
            video_id = video_data_to_process.get('id') # ID видео
            uploader = video_data_to_process.get('uploader', video_data_to_process.get('channel', "Unknown Uploader"))
            logger.info(f"Found YouTube result for '{query}': '{title}' (ID: {video_id}), Uploader: {uploader}, URL: {audio_url[:70]}...")
            return {
                "title": title, 
                "url": audio_url, 
                "uploader": uploader, 
                "video_id": video_id
            }
        else:
            logger.warning(f"Could not extract a playable audio URL for '{query}' from processed video data.")
            return None

    except asyncio.TimeoutError: logger.error(f"yt-dlp command timed out for query: {query}"); return None
    except json.JSONDecodeError as e: logger.error(f"Failed to parse JSON from yt-dlp for '{query}': {e}", exc_info=True); return None
    except FileNotFoundError: logger.error("yt-dlp command not found. Is it installed and in PATH?"); return None
    except Exception as e: logger.error(f"Unexpected error searching YouTube with yt-dlp for '{query}': {e}", exc_info=True); return None

def _update_current_playing_info(source: str, identifier: str, title: Optional[str] = None,
                                artist: Optional[str] = None, uploader: Optional[str] = None):
    global _CURRENTLY_PLAYING_INFO
    _CURRENTLY_PLAYING_INFO = {"source": source, "identifier": identifier, "title": title, "artist": artist, "uploader": uploader}
    logger.debug(f"Updated currently playing info: {_CURRENTLY_PLAYING_INFO}")


# --- Инструменты ---
# play_music, play_from_youtube, pause_music, resume_music, stop_music,
# next_song, previous_song, set_volume, get_current_song
# остаются такими же, как в вашем предыдущем полном примере,
# НО ИЗМЕНЕНИЯ ВНУТРИ like_current_song, unlike_current_song, play_liked_songs, list_liked_songs
# для вызова функций из utils.music_db

@tool
async def play_music(
    song_title: Optional[str] = None, artist_name: Optional[str] = None, album_name: Optional[str] = None,
    playlist_name: Optional[str] = None, search_query: Optional[str] = None, source: Optional[str] = "local"
) -> str:
    """
    Plays music. Can play a specific song, artist, album, or playlist from local library,
    or search and play from YouTube (by setting source to 'youtube' or 'internet').
    If no specific parameters are given and source is 'local', it attempts to resume playback if paused,
    otherwise it plays random songs from the local library.
    Parameters:
        song_title (Optional[str]): The title of the song to play.
        artist_name (Optional[str]): The name of the artist.
        album_name (Optional[str]): The name of the album.
        playlist_name (Optional[str]): The name of a local MPD playlist to load and play.
        search_query (Optional[str]): A general search query for local files or YouTube.
        source (Optional[str]): 'local' (default) or 'youtube'/'internet' to specify search domain.
    """
    logger.info(f"Play music request: song='{song_title}', artist='{artist_name}', album='{album_name}', "
                f"playlist='{playlist_name}', query='{search_query}', source='{source}'")

    # --- Обработка YouTube источника (без изменений) ---
    if source and source.lower() in ["youtube", "internet", "ютуб", "интернет"]:
        query_for_youtube = search_query
        if not query_for_youtube:
            parts = [part for part in [artist_name, album_name, song_title] if part]
            if not parts: return "Please specify what you want to play from YouTube."
            query_for_youtube = " ".join(parts)
        return await play_from_youtube(query_for_youtube)

    # --- Обработка локального плейлиста MPD (без изменений) ---
    if playlist_name:
        _run_mpc_command(['clear'])
        stdout_load, err_load = _run_mpc_command(['load', playlist_name])
        if err_load or not stdout_load: return f"Error loading playlist '{playlist_name}': {err_load if err_load else 'Not found or empty'}."
        stdout_play, err_play = _run_mpc_command(['play'])
        if err_play: return f"Error starting playback for playlist '{playlist_name}': {err_play}"
        # Обновляем информацию о текущем треке (первом из плейлиста)
        await asyncio.sleep(0.3) 
        current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
        if current_song_stdout:
            parts = current_song_stdout.split('\t', 2)
            file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
            mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
            _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
        else: # Плейлист мог быть загружен, но пуст, или mpc current не сработал
             _update_current_playing_info(source="local_mpd", identifier=f"playlist:{playlist_name}", title=f"Playlist {playlist_name}")
        return f"Playing playlist '{playlist_name}'."

    # --- Обработка локального поиска по критериям (без изменений) ---
    search_terms_mpc = []; search_terms_display = []
    if artist_name: search_terms_mpc.extend(['artist', artist_name]); search_terms_display.append(f"artist '{artist_name}'")
    if album_name: search_terms_mpc.extend(['album', album_name]); search_terms_display.append(f"album '{album_name}'")
    if song_title: search_terms_mpc.extend(['title', song_title]); search_terms_display.append(f"song '{song_title}'")
    if not search_terms_mpc and search_query: # search_query используется только если другие поля пусты
        search_terms_mpc.extend(['any', search_query]); search_terms_display.append(f"query '{search_query}'")
    
    if search_terms_mpc: # Если были заданы критерии для локального поиска
        _run_mpc_command(['clear'])
        cmd_to_run = ['findadd'] + search_terms_mpc
        logger.info(f"Running MPC command: mpc {' '.join(cmd_to_run)}")
        stdout_findadd, err_findadd = _run_mpc_command(cmd_to_run)
        # ... (остальная логика обработки результатов findadd и play как раньше) ...
        if err_findadd:
            stdout_search, _ = _run_mpc_command(['search'] + search_terms_mpc)
            if not stdout_search: return f"Couldn't find local music: {', '.join(search_terms_display)}."
            return f"Found matches for {', '.join(search_terms_display)}, but error adding: {err_findadd}"
        playlist_status, _ = _run_mpc_command(['playlist'])
        if not playlist_status: return f"Couldn't find local music: {', '.join(search_terms_display)}."
        stdout_play, err_play = _run_mpc_command(['play'])
        if err_play: return f"Error starting local playback: {err_play}"
        await asyncio.sleep(0.3)
        current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
        playing_what = ""
        if current_song_stdout:
            parts = current_song_stdout.split('\t', 2)
            file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
            mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
            _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
            playing_what = f"Now playing: {mpd_artist + ' - ' if mpd_artist else ''}{mpd_title}"
        else: playing_what = "the first track."
        return f"Playing local music for {', '.join(search_terms_display)}. {playing_what}"
    
    # --- НОВАЯ ЛОГИКА: Нет конкретных критериев (song_title, artist, album, playlist, search_query не заданы, source="local") ---
    else:
        logger.info("No specific criteria for play_music (local source). Checking status or playing random.")
        stdout_status_before_play, _ = _run_mpc_command(['status'])
        
        # Проверяем, есть ли что-то на паузе и можно ли это возобновить
        is_paused_with_song = False
        if stdout_status_before_play:
            if " [paused] " in stdout_status_before_play:
                # Дополнительно проверим, есть ли текущий трек, чтобы не возобновлять пустой плейлист на паузе
                current_check_stdout, _ = _run_mpc_command(['current'])
                if current_check_stdout:
                    is_paused_with_song = True
        
        if is_paused_with_song:
            logger.info("Resuming paused local music.")
            stdout_play, err_play = _run_mpc_command(['play'])
            if err_play: return f"Error resuming local music: {err_play}."
            
            await asyncio.sleep(0.3)
            current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
            if current_song_stdout:
                parts = current_song_stdout.split('\t', 2)
                file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
                mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
                _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
                return f"Resuming playback. Now playing: {mpd_artist + ' - ' if mpd_artist else ''}{mpd_title}."
            else:
                return "Resuming playback." # MPD возобновил, но current не вернул инфо
        else:
            # Плеер не был на паузе (или был на паузе, но плейлист пуст), или был остановлен.
            # Включаем случайную музыку из всей библиотеки.
            logger.info("Playing random local music.")
            _run_mpc_command(['clear'])
            # Добавляем все треки из библиотеки. '/' обычно означает корень музыкальной директории MPD.
            stdout_add_all, err_add_all = _run_mpc_command(['add', '/']) 
            if err_add_all or not _run_mpc_command(['playlist'])[0]: # Проверяем, что плейлист не пуст
                logger.error(f"Failed to add all local tracks to playlist or library is empty. Error: {err_add_all}")
                # Попробуем предложить поиск на YouTube, если локальная библиотека пуста
                return "Your local music library seems empty or I couldn't load it. Would you like me to search on YouTube instead?"

            _run_mpc_command(['shuffle']) # Перемешиваем плейлист
            # _run_mpc_command(['random', 'on']) # Включаем режим случайного воспроизведения (mpc random on)
                                              # 'shuffle' обычно достаточно для одного прохода по перемешанному плейлисту.
                                              # 'random on' будет играть случайные треки из текущего плейлиста постоянно.
                                              # Давайте пока остановимся на shuffle.

            stdout_play, err_play = _run_mpc_command(['play'])
            if err_play: 
                return f"Error starting random local music playback: {err_play}"
            
            await asyncio.sleep(0.3)
            current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
            if current_song_stdout:
                parts = current_song_stdout.split('\t', 2)
                file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
                mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
                _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
                return f"Okay, playing some random music for you! Now playing: {mpd_artist + ' - ' if mpd_artist else ''}{mpd_title}."
            else:
                # Этого не должно произойти, если мы успешно добавили треки и нажали play
                return "Okay, playing some random music for you!"

@tool
async def play_from_youtube(search_query: str) -> str:
    """
    Searches for a song or video on YouTube using the provided query and plays the audio from the first result.
    Parameters:
        search_query (str): The search term (e.g., song title, artist, or video name) for YouTube.
    """
    logger.info(f"Attempting to play from YouTube: '{search_query}'")
    if not search_query: return "Please specify YouTube search query."
    youtube_info = await _search_youtube_and_get_info(search_query)
    if not youtube_info or not youtube_info.get("url"):
        return f"Sorry, couldn't find playable '{search_query}' on YouTube."
    audio_url = youtube_info["url"]; title = youtube_info.get("title", search_query)
    uploader = youtube_info.get("uploader", "Unknown Artist"); video_id = youtube_info.get("video_id", audio_url) # video_id as identifier
    logger.info(f"Adding YouTube stream to MPD: {title} ({audio_url[:60]}...)")
    _run_mpc_command(['clear'])
    stdout_add, err_add = _run_mpc_command(['add', audio_url])
    if err_add:
        logger.error(f"MPC error adding YouTube URL '{audio_url}': {err_add}")
        if "Unsupported URI scheme" in err_add: return f"Found '{title}', but player couldn't handle stream URL."
        return f"Found '{title}', but error adding it: {err_add}."
    stdout_play, err_play = _run_mpc_command(['play'])
    if err_play: return f"Added '{title}', but couldn't start playback: {err_play}"
    _update_current_playing_info(source="youtube", identifier=video_id, title=title, uploader=uploader)
    return f"Okay, playing '{title}' by {uploader} from YouTube."

# --- pause_music, resume_music, stop_music, next_song, previous_song, set_volume, get_current_song ---
# Эти инструменты остаются такими же, но resume_music и get_current_song могут вызывать _update_current_playing_info,
# если они успешно определяют текущий трек. stop_music должен сбрасывать _CURRENTLY_PLAYING_INFO.

@tool
async def pause_music() -> str:
    """Pauses the currently playing music."""
    logger.info("Pausing music."); stdout, err = _run_mpc_command(['pause'])
    return f"Error pausing: {err}" if err else "Music paused."

@tool
async def resume_music() -> str:
    """Resumes playback of paused music. If music was not paused, it may start playing the current queue."""
    logger.info("Resuming music."); stdout_play, err_play = _run_mpc_command(['play'])
    if err_play: return f"Error resuming: {err_play}"
    await asyncio.sleep(0.3)
    current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
    if current_song_stdout:
        parts = current_song_stdout.split('\t', 2)
        file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
        mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
        # Предполагаем, что если что-то играет, это локальный файл. Для YouTube URL это будет сложнее.
        # Если предыдущий трек был с YouTube, _CURRENTLY_PLAYING_INFO уже должно быть установлено play_from_youtube.
        # Здесь мы обновляем только если это локальный файл, идентифицированный mpc.
        if not _CURRENTLY_PLAYING_INFO or _CURRENTLY_PLAYING_INFO.get("source") == "local_mpd":
             _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
        return f"Resuming. Now playing: {mpd_artist + ' - ' if mpd_artist else ''}{mpd_title}."
    return "Resuming playback."


@tool
async def stop_music() -> str:
    """Stops the music playback completely and clears the current playlist."""
    global _CURRENTLY_PLAYING_INFO
    logger.info("Stopping music and clearing playlist.")
    _run_mpc_command(['stop'])
    stdout_clear, err_clear = _run_mpc_command(['clear'])
    _CURRENTLY_PLAYING_INFO = None # Сброс информации о текущем треке
    if err_clear: return f"Music stopped, but error clearing playlist: {err_clear}"
    return "Music stopped and playlist cleared."

@tool
async def next_song() -> str:
    """Skips to and plays the next song in the current playlist."""
    logger.info("Playing next song."); stdout, err = _run_mpc_command(['next'])
    if err: return f"Error playing next: {err}"
    await asyncio.sleep(0.3)
    current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
    if current_song_stdout:
        parts = current_song_stdout.split('\t', 2)
        file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
        mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
        # Опять же, это предполагает локальный трек. Для YouTube плейлиста это будет URL.
        # _update_current_playing_info здесь будет сложнее для смешанных плейлистов.
        # Пока оставим как есть, основной источник инфо - play_music/play_from_youtube.
        # Если плейлист состоит из YouTube URL, то mpc current вернет URL.
        current_identifier = _run_mpc_command(['current', '-f', '%file%'])[0]
        if current_identifier:
            if "youtube.com" in current_identifier or "youtu.be" in current_identifier:
                # Это YouTube трек, но у нас нет video_id и title/uploader из mpc current.
                # _CURRENTLY_PLAYING_INFO не будет точным для лайка, если плейлист смешанный
                # и переключились на YouTube трек из лайкнутых.
                # Для простоты, если это URL, не пытаемся парсить title/artist из него.
                _update_current_playing_info(source="youtube", identifier=current_identifier, title="YouTube Stream")
            else: # Локальный файл
                _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)

        return f"Next: {mpd_title or current_identifier if current_song_stdout else 'Next song (or end of playlist)'}."
    return "Playing next (or end of playlist)."


@tool
async def previous_song() -> str:
    """Goes back to and plays the previous song in the current playlist."""
    logger.info("Playing previous song."); stdout, err = _run_mpc_command(['prev'])
    if err: return f"Error playing previous: {err}"
    await asyncio.sleep(0.3)
    current_song_stdout, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%'])
    if current_song_stdout:
        parts = current_song_stdout.split('\t', 2)
        file_path = parts[0]; mpd_artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
        mpd_title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path
        current_identifier = _run_mpc_command(['current', '-f', '%file%'])[0]
        if current_identifier:
            if "youtube.com" in current_identifier or "youtu.be" in current_identifier:
                _update_current_playing_info(source="youtube", identifier=current_identifier, title="YouTube Stream")
            else:
                _update_current_playing_info(source="local_mpd", identifier=file_path, title=mpd_title, artist=mpd_artist)
        return f"Previous: {mpd_title or current_identifier if current_song_stdout else 'Previous song'}."
    return "Playing previous song."

@tool
async def set_volume(level: Optional[int] = None, change: Optional[str] = None) -> str:
    """
    Sets the music player volume.
    Parameters:
        level (Optional[int]): A specific volume level between 0 and 100.
        change (Optional[str]): A relative change command like '+10', '-10', 'louder', 'quieter', 'mute', or 'max'.
    """
    if level is not None:
        if not 0 <= level <= 100: return "Error: Volume level must be between 0 and 100."
        logger.info(f"Setting volume to {level}%."); stdout, err = _run_mpc_command(['volume', str(level)])
        return f"Error setting volume: {err}" if err else f"Volume set to {level}%."
    elif change:
        change_lower = change.lower(); vol_cmd = None
        if change_lower.startswith(('+', '-')) and change_lower[1:].isdigit(): vol_cmd = change
        elif change_lower in ['louder', 'громче', 'погромче']: vol_cmd = '+10'
        elif change_lower in ['quieter', 'тише', 'потише']: vol_cmd = '-10'
        elif change_lower == 'mute': vol_cmd = '0'
        elif change_lower == 'max': vol_cmd = '100'
        if vol_cmd:
            logger.info(f"Adjusting volume with command: {vol_cmd}"); stdout, err = _run_mpc_command(['volume', vol_cmd])
            if err: return f"Error adjusting volume: {err}"
            await asyncio.sleep(0.1); stdout_status, _ = _run_mpc_command(['status'])
            if stdout_status:
                match = re.search(r"volume:\s*(\d+)%", stdout_status)
                if match: return f"Volume adjusted. Current volume is {match.group(1)}%."
            return "Volume adjusted."
        else: return f"Error: Unknown volume change command '{change}'."
    else: return "Error: Please specify a volume level or a change amount."


@tool
async def get_current_song() -> str:
    """
    Gets information about the currently playing song, including title, artist, album (if available),
    player status (playing/paused/stopped), and current volume level.
    """
    logger.info("Getting current song info.")
    stdout_current_full, _ = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%\\t%album%'])
    stdout_status, _ = _run_mpc_command(['status'])
    response_parts = []

    if stdout_current_full:
        parts = stdout_current_full.split('\t', 3)
        file_path = parts[0]
        artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
        title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else file_path # Fallback to file/URL
        album = parts[3] if len(parts) > 3 and parts[3] != '(null)' else None

        display_title = title
        if artist: display_title = f"{artist} - {title}"
        if album: display_title += f" (Album: {album})"
        response_parts.append(f"Playing: {display_title}")

        # Обновляем _CURRENTLY_PLAYING_INFO
        if "youtube.com" in file_path or "youtu.be" in file_path:
            # Для YouTube, file_path это URL. Нам нужен video_id для надежного identifier.
            # Это сложно получить из mpc current. _CURRENTLY_PLAYING_INFO для YouTube
            # должно было быть установлено при вызове play_from_youtube.
            # Если текущий трек - YouTube URL, и _CURRENTLY_PLAYING_INFO не соответствует,
            # то лайк может сохранить URL вместо video_id. Это ограничение.
            if not (_CURRENTLY_PLAYING_INFO and _CURRENTLY_PLAYING_INFO.get("identifier") in file_path and _CURRENTLY_PLAYING_INFO.get("source") == "youtube"):
                 _update_current_playing_info(source="youtube", identifier=file_path, title=title or "YouTube Stream", uploader=artist) # Uploader может быть artist
        else:
            _update_current_playing_info(source="local_mpd", identifier=file_path, title=title, artist=artist)
    else:
        response_parts.append("Nothing is currently playing.")
        _CURRENTLY_PLAYING_INFO = None # Сброс, если ничего не играет

    if stdout_status:
        state_match = re.search(r"\[(playing|paused|stopped)\]", stdout_status)
        volume_match = re.search(r"volume:\s*(\d+)%", stdout_status)
        if state_match: response_parts.append(f"Status: {state_match.group(1)}")
        if volume_match: response_parts.append(f"Volume: {volume_match.group(1)}%")

    return ". ".join(response_parts) + "." if response_parts else "Could not get player status."


# --- Новые инструменты для "лайков" ---
@tool
async def like_current_song() -> str:
    """
    Adds the currently playing song to your liked songs list.
    It attempts to identify the song from player status. If successful, the song (local file path or YouTube video ID)
    along with its available metadata (title, artist/uploader) is saved.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        # Попробуем получить информацию о текущем треке, если _CURRENTLY_PLAYING_INFO пусто
        await get_current_song() # Это обновит _CURRENTLY_PLAYING_INFO, если что-то играет
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to like it."

    source = _CURRENTLY_PLAYING_INFO["source"]
    identifier = _CURRENTLY_PLAYING_INFO["identifier"]
    title = _CURRENTLY_PLAYING_INFO.get("title")
    artist = _CURRENTLY_PLAYING_INFO.get("artist")
    uploader = _CURRENTLY_PLAYING_INFO.get("uploader")

    # Если это YouTube, и identifier это URL, а не video_id, это проблема для долгосрочного хранения.
    # _search_youtube_and_get_info теперь возвращает video_id, и play_from_youtube должен его использовать.
    if source == "youtube" and ("youtube.com" in identifier or "youtu.be" in identifier) and not title:
        # Если identifier - это URL, а title нет, значит, _CURRENTLY_PLAYING_INFO неполное.
        # Этого не должно происходить, если play_from_youtube правильно установил video_id.
        logger.warning(f"Attempting to like a YouTube stream with URL as identifier and no title: {identifier}")
        # Можно попробовать извлечь title из URL или просто сохранить URL как есть, но это менее надежно.
        # Для простоты, пока разрешим, но это точка для улучшения.
        if not title: title = "YouTube Stream"


    if await is_song_liked_in_db(source, identifier): # Используем функцию из utils.music_db
        return f"You've already liked '{title or identifier}'."

    success = await add_song_to_liked_db(source, identifier, title, artist, uploader) # Из utils.music_db
    if success:
        return f"Okay, I've added '{title or identifier}' to your liked songs."
    else:
        return f"Sorry, I couldn't add '{title or identifier}' to your liked songs."

@tool
async def unlike_current_song() -> str:
    """
    Removes the currently playing song from your liked songs list.
    It identifies the song from player status and removes its entry from the liked songs database.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        await get_current_song()
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to unlike it."

    source = _CURRENTLY_PLAYING_INFO["source"]
    identifier = _CURRENTLY_PLAYING_INFO["identifier"]
    title = _CURRENTLY_PLAYING_INFO.get("title", identifier)

    if not await is_song_liked_in_db(source, identifier): # Из utils.music_db
        return f"It seems '{title}' wasn't in your liked songs list."

    success = await remove_song_from_liked_db(source, identifier) # Из utils.music_db
    if success:
        return f"Okay, I've removed '{title}' from your liked songs."
    else:
        return f"Sorry, I couldn't remove '{title}' from your liked songs."

@tool
async def play_liked_songs(shuffle: Optional[bool] = False) -> str:
    """
    Plays songs from your liked songs list. Clears the current queue and adds all liked songs.
    For YouTube songs, it attempts to fetch fresh stream URLs.
    Parameters:
        shuffle (Optional[bool]): If True, shuffles the liked songs playlist before playing. Defaults to False.
    """
    logger.info(f"Playing liked songs. Shuffle: {shuffle}")
    liked_songs = await get_all_liked_songs_from_db() # Из utils.music_db

    if not liked_songs:
        return "You don't have any liked songs yet!"

    _run_mpc_command(['clear'])
    added_count = 0
    tracks_to_play_info = [] # Для обновления _CURRENTLY_PLAYING_INFO для первого трека

    for song_data in liked_songs:
        identifier = song_data['identifier']
        playable_identifier = identifier
        current_track_info_for_update = song_data.copy() # Копируем для _CURRENTLY_PLAYING_INFO

        if song_data['source'] == 'youtube':
            logger.info(f"Liked song is YouTube (ID/Stored Identifier: {identifier}). Getting fresh stream URL.")
            # Identifier для YouTube должен быть video_id
            # Если title нет, используем video_id для поиска (менее надежно)
            yt_query = song_data.get('title') if song_data.get('title') else \
                       (identifier if not ("youtube.com" in identifier or "youtu.be" in identifier) else "previously liked youtube video")

            youtube_info = await _search_youtube_and_get_info(yt_query)
            if youtube_info and youtube_info.get("url"):
                playable_identifier = youtube_info["url"]
                # Обновляем информацию для _CURRENTLY_PLAYING_INFO
                current_track_info_for_update['title'] = youtube_info.get('title', song_data.get('title'))
                current_track_info_for_update['uploader'] = youtube_info.get('uploader', song_data.get('uploader'))
                current_track_info_for_update['identifier'] = youtube_info.get('video_id', identifier) # Убеждаемся, что identifier это video_id
            else:
                logger.warning(f"Could not get playable URL for liked YouTube song: ID/Query '{identifier}' / '{yt_query}'")
                continue

        _stdout, err = _run_mpc_command(['add', playable_identifier])
        if not err:
            added_count += 1
            if not tracks_to_play_info: # Сохраняем инфо о первом добавленном треке
                tracks_to_play_info.append(current_track_info_for_update)
        else:
            logger.warning(f"Failed to add liked song '{playable_identifier}' to MPD: {err}")

    if added_count == 0:
        return "Found liked songs, but couldn't add any to the player."

    if shuffle: _run_mpc_command(['shuffle'])
    _stdout_play, err_play = _run_mpc_command(['play'])
    if err_play: return f"Added {added_count} liked songs, but couldn't start playback: {err_play}"

    await asyncio.sleep(0.3)
    current_song_mpc, _ = _run_mpc_command(['current'])

    # Обновляем _CURRENTLY_PLAYING_INFO для первого трека в плейлисте лайков
    if tracks_to_play_info:
        first_track = tracks_to_play_info[0]
        _update_current_playing_info(
            source=first_track['source'],
            identifier=first_track['identifier'], # Должен быть video_id для YouTube
            title=first_track.get('title'),
            artist=first_track.get('artist'),
            uploader=first_track.get('uploader')
        )

    playing_what = f"Now playing: {current_song_mpc}." if current_song_mpc else "Starting with the first liked song."
    return f"Okay, playing {added_count} liked song(s). {playing_what}"

@tool
async def list_liked_songs() -> str:
    """
    Lists all your liked songs from the database.
    Displays the title, artist/uploader, and source (Local or YouTube) for each liked song.
    Songs are listed newest first.
    """
    liked_songs = await get_all_liked_songs_from_db()
    if not liked_songs: return "You haven't liked any songs yet."
    response_lines = ["Here are your liked songs (newest first):"]
    for i, song_data in enumerate(liked_songs):
        title = song_data.get('title', song_data.get('identifier', 'Unknown Title')) # Используем identifier если title пуст
        artist = song_data.get('artist')
        uploader = song_data.get('uploader')
        source_type = "Local" if song_data['source'] == 'local_mpd' else "YouTube"
        display_name = title
        if artist and source_type == "Local": display_name = f"{artist} - {title}"
        elif uploader and source_type == "YouTube": display_name = f"{title} (by {uploader})"
        response_lines.append(f"{i+1}. {display_name} [{source_type}]") # ID из БД не показываем пользователю
    return "\n".join(response_lines)