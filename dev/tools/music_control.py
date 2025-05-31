import asyncio
import subprocess
import logging
from typing import Optional, List, Tuple, Dict, Any
import re
import json
import random # Для случайной выборки треков
import time   # Для TTL кэша YouTube

from langchain_core.tools import tool
from settings import settings # Для DB_PATH и других настроек, если понадобятся

from utils.music_db import (
    add_song_to_liked_db,
    remove_song_from_liked_db,
    get_all_liked_songs_from_db,
    is_song_liked_in_db
)

logger = logging.getLogger(__name__)

_CURRENTLY_PLAYING_INFO: Optional[Dict[str, Any]] = None
_ORIGINAL_MPD_VOLUME_BEFORE_DUCKING: Optional[int] = None

# --- Кэш для результатов YouTube поиска ---
YT_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
YT_CACHE_TTL_SECONDS = settings.music_control.get("youtube_cache_ttl_seconds", 3600) if hasattr(settings, 'music_control') and isinstance(settings.music_control, dict) else 3600
# Структура YT_SEARCH_CACHE[query] = {"timestamp": time.time(), "data": youtube_info}


# --- Вспомогательные функции для MPC ---

def _run_mpc_command(args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        # Увеличим таймаут для команд, которые могут быть долгими (например, listall на большой библиотеке)
        timeout_duration = 30 if args[0] in ['listall', 'add /'] else 10
        process = subprocess.Popen(['mpc'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=timeout_duration)
        is_non_error_empty_stdout = args[0] in ['current', 'playlist'] and not stdout.strip() and not stderr.strip()
        
        if process.returncode != 0 and not is_non_error_empty_stdout:
            err_msg = stderr.strip() if stderr else f"MPC command failed with code {process.returncode}"
            logger.warning(f"MPC command {' '.join(args)} error (code {process.returncode}): {err_msg}. Stdout: {stdout.strip() if stdout else 'empty'}")
            return stdout.strip() if stdout else None, err_msg
        
        if stderr and stderr.strip() and process.returncode == 0:
             logger.info(f"MPC command {' '.join(args)} successful with stderr info: {stderr.strip()}")

        return stdout.strip() if stdout else None, None
    except FileNotFoundError: logger.error("MPC command not found."); return None, "MPC command not found."
    except subprocess.TimeoutExpired: logger.error(f"MPC command {' '.join(args)} timed out."); return None, "MPC command timed out."
    except Exception as e: logger.error(f"Error running MPC command {' '.join(args)}: {e}", exc_info=True); return None, f"Unexpected MPC error: {str(e)}"


async def _get_current_mpd_track_details() -> Optional[Dict[str, Any]]:
    stdout_current, err_current = _run_mpc_command(['current', '-f', '%file%\\t%artist%\\t%title%\\t%album%'])
    if err_current or not stdout_current:
        status_out, _ = _run_mpc_command(['status'])
        if status_out and ("volume:" in status_out and not "[playing]" in status_out and not "[paused]" in status_out):
            logger.debug("MPD: Nothing seems to be playing (checked via status after empty 'current').")
            return None
        elif err_current:
            logger.warning(f"Could not get current MPD track details: {err_current}")
        return None

    parts = stdout_current.split('\t', 3)
    file_path = parts[0]
    artist = parts[1] if len(parts) > 1 and parts[1] != '(null)' else None
    title = parts[2] if len(parts) > 2 and parts[2] != '(null)' else None
    album = parts[3] if len(parts) > 3 and parts[3] != '(null)' else None
    
    if not title and file_path and not ("http://" in file_path or "https://" in file_path):
        try:
            title = file_path.split('/')[-1].rsplit('.', 1)[0] # Имя файла без расширения
        except:
            title = file_path

    return {"file": file_path, "artist": artist, "title": title, "album": album}

async def _update_currently_playing_info_from_mpd() -> bool:
    global _CURRENTLY_PLAYING_INFO
    mpd_details = await _get_current_mpd_track_details()

    if not mpd_details or not mpd_details.get("file"):
        _CURRENTLY_PLAYING_INFO = None
        logger.debug("_update_currently_playing_info_from_mpd: No track playing in MPD or no file info.")
        return False

    file_path = mpd_details["file"]
    artist = mpd_details["artist"]
    title = mpd_details["title"]

    if "youtube.com/" in file_path or "youtu.be/" in file_path:
        logger.info(f"MPD is playing a YouTube URL: {file_path}. Attempting to get video details.")
        youtube_info = await _search_youtube_and_get_info(file_path)
        if youtube_info and youtube_info.get("video_id"):
            _CURRENTLY_PLAYING_INFO = {
                "source": "youtube",
                "identifier": youtube_info["video_id"],
                "title": youtube_info.get("title", title or "YouTube Video"),
                "uploader": youtube_info.get("uploader", artist),
                "url": file_path 
            }
            logger.debug(f"Updated playing info for YouTube (from MPD): {_CURRENTLY_PLAYING_INFO}")
            return True
        else:
            _CURRENTLY_PLAYING_INFO = {
                "source": "youtube",
                "identifier": file_path, 
                "title": title or "YouTube Stream",
                "uploader": artist,
                "url": file_path
            }
            logger.warning(f"Could not get video_id for YouTube URL from MPD: {file_path}. Using URL as identifier.")
            return True
    else:
        _CURRENTLY_PLAYING_INFO = {
            "source": "local_mpd",
            "identifier": file_path,
            "title": title or file_path.split('/')[-1],
            "artist": artist,
            "album": mpd_details.get("album")
        }
        logger.debug(f"Updated playing info for local_mpd: {_CURRENTLY_PLAYING_INFO}")
        return True
    return False


def _update_current_playing_info(source: str, identifier: str, title: Optional[str] = None,
                                artist: Optional[str] = None, uploader: Optional[str] = None,
                                album: Optional[str] = None, url: Optional[str] = None):
    global _CURRENTLY_PLAYING_INFO
    _CURRENTLY_PLAYING_INFO = {
        "source": source, "identifier": identifier, "title": title,
        "artist": artist, "uploader": uploader, "album": album, "url": url
    }
    logger.debug(f"Set currently playing info: {_CURRENTLY_PLAYING_INFO}")


async def _search_youtube_and_get_info(query: str) -> Optional[Dict[str, Any]]:
    cached_entry = YT_SEARCH_CACHE.get(query)
    if cached_entry:
        if (time.time() - cached_entry["timestamp"]) < YT_CACHE_TTL_SECONDS:
            logger.info(f"Cache hit for YouTube query: '{query}'")
            return cached_entry["data"]
        else:
            logger.info(f"Cache expired for YouTube query: '{query}'")
            del YT_SEARCH_CACHE[query]

    logger.info(f"Searching YouTube with yt-dlp for: '{query}' (cache miss or expired)")
    is_direct_url = "youtube.com/" in query or "youtu.be/" in query
    yt_dlp_args = [
        'yt-dlp', '--skip-download', '--dump-single-json',
        '--ignore-errors', '--no-warnings', '--format', 'bestaudio/best',
    ]
    if not is_direct_url:
        yt_dlp_args.extend(['--default-search', 'ytsearch1:'])
    yt_dlp_args.append(query)

    try:
        process = await asyncio.create_subprocess_exec(
            *yt_dlp_args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)
        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()

        if process.returncode != 0:
            logger.error(f"yt-dlp failed for '{query}' (code {process.returncode}). Stderr: {stderr_str}")
            if "Video unavailable" in stderr_str: logger.warning(f"yt-dlp: Video '{query}' is unavailable.")
            elif "age restricted" in stderr_str.lower(): logger.warning(f"yt-dlp: Video '{query}' is age-restricted.")
            return None
        
        if not stdout_bytes:
            logger.warning(f"yt-dlp returned no stdout for '{query}'. Stderr: {stderr_str}"); return None

        result_info = json.loads(stdout_bytes.decode('utf-8', errors='ignore'))
        video_data_to_process = result_info["entries"][0] if result_info.get("_type") == "playlist" and "entries" in result_info and result_info["entries"] else result_info

        if not video_data_to_process:
            logger.warning(f"No video data to process from yt-dlp output for '{query}'."); return None

        title_from_yt = video_data_to_process.get('title', query) # Используем query как fallback для title
        audio_url = video_data_to_process.get('url')
        
        if not audio_url and 'formats' in video_data_to_process:
            logger.debug(f"No top-level 'url' in video data for '{title_from_yt}', checking formats list...")
            audio_formats = [f for f in video_data_to_process['formats'] if f.get('url') and f.get('acodec') and f.get('acodec') != 'none']
            if audio_formats:
                only_audio = [f for f in audio_formats if f.get('vcodec') == 'none']
                target_list = only_audio if only_audio else audio_formats
                preferred_exts = ('m4a', 'opus', 'webm', 'mp3')
                for ext in preferred_exts:
                    for f_info in target_list:
                        if f_info.get('ext') == ext: audio_url = f_info['url']; break
                    if audio_url: break
                if not audio_url and target_list: audio_url = target_list[0]['url']

        if audio_url:
            video_id = video_data_to_process.get('id')
            uploader = video_data_to_process.get('uploader', video_data_to_process.get('channel', "Unknown Uploader"))
            duration = video_data_to_process.get('duration')
            
            youtube_result_data = {
                "title": title_from_yt, "url": audio_url, "uploader": uploader, 
                "video_id": video_id, "duration": duration
            }
            YT_SEARCH_CACHE[query] = {"timestamp": time.time(), "data": youtube_result_data}
            logger.info(f"Found and cached YouTube result for query: '{query}'")
            return youtube_result_data
        else:
            logger.warning(f"Could not extract a playable audio URL for '{query}' from processed video data. Not caching.")
            return None

    except asyncio.TimeoutError: logger.error(f"yt-dlp command timed out for query: {query}"); return None
    except json.JSONDecodeError as e: logger.error(f"Failed to parse JSON from yt-dlp for '{query}': {e}", exc_info=True); return None
    except FileNotFoundError: logger.error("yt-dlp command not found. Is it installed and in PATH?"); return None
    except Exception as e: logger.error(f"Unexpected error searching YouTube with yt-dlp for '{query}': {e}", exc_info=True); return None


# --- Инструменты ---

@tool
async def play_music(
    song_title: Optional[str] = None, artist_name: Optional[str] = None, album_name: Optional[str] = None,
    playlist_name: Optional[str] = None, search_query: Optional[str] = None, source: Optional[str] = "local",
    random_sample_size: int = settings.music_control.get("random_sample_size", 100) if hasattr(settings, 'music_control') and isinstance(settings.music_control, dict) else 100
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
                f"playlist='{playlist_name}', query='{search_query}', source='{source}', random_sample_size='{random_sample_size}'")

    if source and source.lower() in ["youtube", "internet", "ютуб", "интернет"]:
        query_for_youtube = search_query
        if not query_for_youtube:
            parts = [part for part in [artist_name, song_title, album_name] if part]
            if not parts: return "Please specify what you want to play from YouTube."
            query_for_youtube = " ".join(parts)
        return await play_from_youtube(query_for_youtube)

    _run_mpc_command(['clear'])

    if playlist_name:
        _, err_load = _run_mpc_command(['load', playlist_name])
        if err_load: return f"Error loading playlist '{playlist_name}': {err_load}."
        playlist_content, _ = _run_mpc_command(['playlist'])
        if not playlist_content: return f"Playlist '{playlist_name}' was loaded but it seems to be empty."
        _, err_play = _run_mpc_command(['play'])
        if err_play: return f"Error starting playback for playlist '{playlist_name}': {err_play}"
        await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
        current_title = _CURRENTLY_PLAYING_INFO.get("title", "first track") if _CURRENTLY_PLAYING_INFO else "first track"
        return f"Playing playlist '{playlist_name}'. Starting with: {current_title}."

    search_terms_mpc = []; search_terms_display = []
    if artist_name: search_terms_mpc.extend(['artist', artist_name]); search_terms_display.append(f"artist '{artist_name}'")
    if album_name: search_terms_mpc.extend(['album', album_name]); search_terms_display.append(f"album '{album_name}'")
    if song_title: search_terms_mpc.extend(['title', song_title]); search_terms_display.append(f"song '{song_title}'")
    if not search_terms_mpc and search_query:
        search_terms_mpc.extend(['any', search_query]); search_terms_display.append(f"query '{search_query}'")
    
    if search_terms_mpc:
        _, err_findadd = _run_mpc_command(['findadd'] + search_terms_mpc)
        if err_findadd:
            search_results, _ = _run_mpc_command(['search'] + search_terms_mpc)
            if not search_results: return f"Couldn't find any local music matching: {', '.join(search_terms_display)}."
            else: return f"Found matches for {', '.join(search_terms_display)}, but encountered an error adding them: {err_findadd}"
        playlist_status, _ = _run_mpc_command(['playlist'])
        if not playlist_status: return f"Couldn't find any local music matching: {', '.join(search_terms_display)} (playlist empty after findadd)."
        _, err_play = _run_mpc_command(['play'])
        if err_play: return f"Error starting local playback: {err_play}"
        await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
        current_title = _CURRENTLY_PLAYING_INFO.get("title", "first track") if _CURRENTLY_PLAYING_INFO else "first track"
        return f"Playing local music for {', '.join(search_terms_display)}. Now playing: {current_title}."
    
    else: 
        logger.info("No specific criteria for play_music (local source). Checking status or playing random.")
        status_stdout, _ = _run_mpc_command(['status'])
        is_paused_with_song = False
        if status_stdout and " [paused] " in status_stdout:
            current_track_details = await _get_current_mpd_track_details()
            if current_track_details and current_track_details.get("file"): is_paused_with_song = True
        if is_paused_with_song: logger.info("Resuming paused local music."); return await resume_music() 
        else:
            logger.info(f"Playing random local music (sample size: {random_sample_size}).")
            all_local_tracks_stdout, err_ls = _run_mpc_command(['listall'])
            if err_ls or not all_local_tracks_stdout:
                logger.error(f"Failed to list all local tracks for random play: {err_ls}. Falling back to 'add /'.")
                _, err_add_all = _run_mpc_command(['add', '/'])
                if err_add_all: return f"Failed to load local music library for random play: {err_add_all}. Is your library configured in MPD?"
            else:
                all_tracks_list = [track for track in all_local_tracks_stdout.splitlines() if track.strip()]
                if not all_tracks_list: return "Your local music library seems empty. Would you like me to search on YouTube instead?"
                tracks_to_add = random.sample(all_tracks_list, min(len(all_tracks_list), random_sample_size))
                logger.info(f"Adding {len(tracks_to_add)} random tracks to the playlist.")
                added_successfully_count = 0
                for track_path in tracks_to_add:
                    _, err_add_track = _run_mpc_command(['add', track_path])
                    if err_add_track: logger.warning(f"Failed to add random track '{track_path}': {err_add_track}")
                    else: added_successfully_count +=1
                if added_successfully_count == 0: return "Could not add any random tracks to the playlist. Please check MPD logs."
            
            playlist_content, _ = _run_mpc_command(['playlist'])
            if not playlist_content: return "Your local music library seems empty or I couldn't load it after sampling. Would you like me to search on YouTube instead?"
            _run_mpc_command(['shuffle'])
            _, err_play = _run_mpc_command(['play'])
            if err_play: return f"Error starting random local music playback: {err_play}"
            await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
            current_title = _CURRENTLY_PLAYING_INFO.get("title", "a random song") if _CURRENTLY_PLAYING_INFO else "a random song"
            return f"Okay, playing some random music for you! Now playing: {current_title}."

@tool
async def play_from_youtube(search_query: str) -> str:
    """
    Searches for a song or video on YouTube using the provided query and plays the audio from the first result.
    Parameters:
        search_query (str): The search term (e.g., song title, artist, or video name) for YouTube.
    """
    logger.info(f"Attempting to play from YouTube: '{search_query}'")
    if not search_query: return "Please specify a YouTube search query."
    youtube_info = await _search_youtube_and_get_info(search_query)
    if not youtube_info or not youtube_info.get("url"):
        return f"Sorry, I couldn't find a playable result for '{search_query}' on YouTube."

    audio_url = youtube_info["url"]; title = youtube_info.get("title", search_query)
    uploader = youtube_info.get("uploader", "Unknown Artist"); video_id = youtube_info.get("video_id", audio_url)
    logger.info(f"Adding YouTube stream to MPD: {title} (ID: {video_id}, URL: {audio_url[:60]}...)")
    _run_mpc_command(['clear'])
    _, err_add = _run_mpc_command(['add', audio_url])
    if err_add:
        logger.error(f"MPC error adding YouTube URL '{audio_url}': {err_add}")
        if "Unsupported URI scheme" in err_add or "Invalid argument" in err_add:
            return f"Found '{title}' on YouTube, but the player couldn't handle its stream URL. It might be a protected stream or an unsupported format."
        return f"Found '{title}' on YouTube, but there was an error adding it to the player: {err_add}."
    _, err_play = _run_mpc_command(['play'])
    if err_play: return f"Added '{title}' to the player, but couldn't start playback: {err_play}"
    _update_current_playing_info(source="youtube", identifier=video_id, title=title, uploader=uploader, url=audio_url)
    return f"Okay, playing '{title}' by {uploader} from YouTube."


@tool
async def pause_music() -> str:
    """Pauses the currently playing music."""
    logger.info("Pausing music.")
    status_before, _ = _run_mpc_command(['status'])
    if not status_before or "[playing]" not in status_before: return "Nothing is currently playing to pause."
    _, err = _run_mpc_command(['pause']); return f"Error pausing: {err}" if err else "Music paused."

@tool
async def resume_music() -> str:
    """Resumes playback of paused music. If music was not paused, it may start playing the current queue."""
    logger.info("Resuming music.")
    status_before, _ = _run_mpc_command(['status'])
    if status_before and "[playing]" in status_before: return "Music is already playing."
    if not status_before or not ("[paused]" in status_before or "volume:" in status_before):
        return "Nothing to resume. The playlist might be empty or stopped. Try 'play music'."
    _, err_play = _run_mpc_command(['play'])
    if err_play: return f"Error resuming: {err_play}"
    await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
    if _CURRENTLY_PLAYING_INFO and _CURRENTLY_PLAYING_INFO.get("title"):
        title = _CURRENTLY_PLAYING_INFO.get("title")
        artist_or_uploader = _CURRENTLY_PLAYING_INFO.get("artist") or _CURRENTLY_PLAYING_INFO.get("uploader")
        display_name = f"{artist_or_uploader} - {title}" if artist_or_uploader else title
        return f"Resuming playback. Now playing: {display_name}."
    elif _CURRENTLY_PLAYING_INFO and _CURRENTLY_PLAYING_INFO.get("identifier"):
        return f"Resuming playback. Now playing: {_CURRENTLY_PLAYING_INFO.get('identifier')}."
    else:
        status_after, _ = _run_mpc_command(['status'])
        if status_after and "[playing]" in status_after: return "Resuming playback."
        else: return "Resumed, but it seems the playlist finished or is empty."


@tool
async def stop_music() -> str:
    """Stops the music playback completely and clears the current playlist."""
    global _CURRENTLY_PLAYING_INFO
    logger.info("Stopping music and clearing playlist.")
    _run_mpc_command(['stop']); _, err_clear = _run_mpc_command(['clear'])
    _CURRENTLY_PLAYING_INFO = None
    if err_clear: return f"Music stopped, but error clearing playlist: {err_clear}"
    return "Music stopped and playlist cleared."

async def _handle_track_change(command: str) -> str:
    logger.info(f"Executing MPC command: {command}"); _, err = _run_mpc_command([command])
    if err: return f"Error executing '{command}': {err}"
    await asyncio.sleep(0.2); updated = await _update_currently_playing_info_from_mpd()
    if updated and _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title")
        artist_or_uploader = _CURRENTLY_PLAYING_INFO.get("artist") or _CURRENTLY_PLAYING_INFO.get("uploader")
        display_name = f"{artist_or_uploader} - {title}" if artist_or_uploader and title else (title or _CURRENTLY_PLAYING_INFO.get("identifier", "next track"))
        return f"{command.capitalize()}: {display_name}."
    else:
        status_after, _ = _run_mpc_command(['status'])
        if status_after and not ("[playing]" in status_after or "[paused]" in status_after):
            return f"{command.capitalize()}: Reached the end of the playlist."
        return f"{command.capitalize()}: Switched track (details unavailable or end of playlist)."

@tool
async def next_song() -> str:
    """Skips to and plays the next song in the current playlist."""
    return await _handle_track_change("next")
@tool
async def previous_song() -> str:
    """Goes back to and plays the previous song in the current playlist."""
    return await _handle_track_change("prev")


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
        logger.info(f"Setting volume to {level}%."); _, err = _run_mpc_command(['volume', str(level)])
        return f"Error setting volume: {err}" if err else f"Volume set to {level}%."
    elif change:
        change_lower = change.lower(); vol_cmd = None
        if change_lower.startswith(('+', '-')) and change_lower[1:].isdigit(): vol_cmd = change
        elif change_lower in ['louder', 'громче', 'погромче']: vol_cmd = '+10'
        elif change_lower in ['quieter', 'тише', 'потише']: vol_cmd = '-10'
        elif change_lower == 'mute': vol_cmd = '0'
        elif change_lower == 'max': vol_cmd = '100'
        if vol_cmd:
            logger.info(f"Adjusting volume with command: {vol_cmd}"); _, err = _run_mpc_command(['volume', vol_cmd])
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
    logger.info("Getting current song info."); await _update_currently_playing_info_from_mpd() 
    response_parts = []
    if _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title")
        artist = _CURRENTLY_PLAYING_INFO.get("artist"); uploader = _CURRENTLY_PLAYING_INFO.get("uploader")
        album = _CURRENTLY_PLAYING_INFO.get("album"); identifier = _CURRENTLY_PLAYING_INFO.get("identifier", "Unknown track")
        display_title = title or identifier
        if artist: display_title = f"{artist} - {display_title}"
        elif uploader: display_title = f"{display_title} (by {uploader})"
        if album: display_title += f" (Album: {album})"
        response_parts.append(f"Currently: {display_title}")
    else: response_parts.append("Nothing is currently playing.")
    stdout_status, _ = _run_mpc_command(['status'])
    if stdout_status:
        state_match = re.search(r"\[(playing|paused|stopped)\]", stdout_status)
        volume_match = re.search(r"volume:\s*(\d+)%", stdout_status)
        if state_match: response_parts.append(f"Status: {state_match.group(1)}")
        if volume_match: response_parts.append(f"Volume: {volume_match.group(1)}%")
    else: response_parts.append("Could not get player status details.")
    return ". ".join(response_parts) + "."


@tool
async def like_current_song() -> str:
    """
    Adds the currently playing song to your liked songs list.
    It attempts to identify the song from player status. If successful, the song (local file path or YouTube video ID)
    along with its available metadata (title, artist/uploader) is saved.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        logger.info("Like request: _CURRENTLY_PLAYING_INFO is empty or lacks identifier. Attempting to update.")
        await _update_currently_playing_info_from_mpd()
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to like it."
    source = _CURRENTLY_PLAYING_INFO["source"]; identifier = _CURRENTLY_PLAYING_INFO["identifier"]
    title = _CURRENTLY_PLAYING_INFO.get("title"); artist = _CURRENTLY_PLAYING_INFO.get("artist")
    uploader = _CURRENTLY_PLAYING_INFO.get("uploader")
    if source == "youtube" and ("youtube.com/" in identifier or "youtu.be/" in identifier):
        logger.warning(f"Attempting to like a YouTube song, but identifier is a URL: {identifier}. This is not ideal.")
        if not title: title = "YouTube Video (URL liked)"
    display_name = title or identifier
    if await is_song_liked_in_db(source, identifier): return f"You've already liked '{display_name}'."
    success = await add_song_to_liked_db(source, identifier, title, artist, uploader)
    if success: return f"Okay, I've added '{display_name}' to your liked songs."
    else: return f"Sorry, I couldn't add '{display_name}' to your liked songs due to a database error."

@tool
async def unlike_current_song() -> str:
    """
    Removes the currently playing song from your liked songs list.
    It identifies the song from player status and removes its entry from the liked songs database.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        logger.info("Unlike request: _CURRENTLY_PLAYING_INFO is empty or lacks identifier. Attempting to update.")
        await _update_currently_playing_info_from_mpd()
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to unlike it."
    source = _CURRENTLY_PLAYING_INFO["source"]; identifier = _CURRENTLY_PLAYING_INFO["identifier"]
    title = _CURRENTLY_PLAYING_INFO.get("title", identifier)
    if not await is_song_liked_in_db(source, identifier): return f"It seems '{title}' wasn't in your liked songs list."
    success = await remove_song_from_liked_db(source, identifier)
    if success: return f"Okay, I've removed '{title}' from your liked songs."
    else: return f"Sorry, I couldn't remove '{title}' from your liked songs due to a database error."

@tool
async def play_liked_songs(shuffle: Optional[bool] = False) -> str:
    """
    Plays songs from your liked songs list. Clears the current queue and adds all liked songs.
    For YouTube songs, it attempts to fetch fresh stream URLs.
    Parameters:
        shuffle (Optional[bool]): If True, shuffles the liked songs playlist before playing. Defaults to False.
    """
    logger.info(f"Playing liked songs. Shuffle: {shuffle}")
    liked_songs = await get_all_liked_songs_from_db()
    if not liked_songs: return "You don't have any liked songs yet!"
    _run_mpc_command(['clear']); added_count = 0
    first_track_to_play_info: Optional[Dict[str, Any]] = None
    for song_data in liked_songs:
        identifier = song_data['identifier']; source = song_data['source']
        title_from_db = song_data.get('title'); uploader_from_db = song_data.get('uploader')
        playable_identifier_for_mpc = identifier; current_track_info_for_update = song_data.copy()
        if source == 'youtube':
            logger.info(f"Liked song is YouTube (ID: {identifier}). Getting fresh stream URL.")
            youtube_info = await _search_youtube_and_get_info(identifier)
            if youtube_info and youtube_info.get("url"):
                playable_identifier_for_mpc = youtube_info["url"]
                current_track_info_for_update['title'] = youtube_info.get('title', title_from_db)
                current_track_info_for_update['uploader'] = youtube_info.get('uploader', uploader_from_db)
                current_track_info_for_update['identifier'] = youtube_info.get('video_id', identifier)
                current_track_info_for_update['url'] = playable_identifier_for_mpc
            else: logger.warning(f"Could not get playable URL for liked YouTube song (ID: '{identifier}'). Skipping."); continue
        _, err = _run_mpc_command(['add', playable_identifier_for_mpc])
        if not err:
            added_count += 1
            if not first_track_to_play_info: first_track_to_play_info = current_track_info_for_update
        else: logger.warning(f"Failed to add liked song '{playable_identifier_for_mpc}' to MPD: {err}")
    if added_count == 0: return "Found liked songs, but couldn't add any to the player (possibly due to unavailable YouTube streams)."
    if shuffle: _run_mpc_command(['shuffle'])
    _, err_play = _run_mpc_command(['play'])
    if err_play: return f"Added {added_count} liked songs, but couldn't start playback: {err_play}"
    if first_track_to_play_info:
        _update_current_playing_info(
            source=first_track_to_play_info['source'], identifier=first_track_to_play_info['identifier'],
            title=first_track_to_play_info.get('title'), artist=first_track_to_play_info.get('artist'),
            uploader=first_track_to_play_info.get('uploader'), album=first_track_to_play_info.get('album'),
            url=first_track_to_play_info.get('url')
        )
        display_name = first_track_to_play_info.get('title', first_track_to_play_info.get('identifier'))
        return f"Okay, playing {added_count} liked song(s). Starting with: {display_name}."
    else: return f"Okay, playing {added_count} liked song(s)."


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
        title = song_data.get('title', song_data.get('identifier', 'Unknown Title')) 
        artist = song_data.get('artist'); uploader = song_data.get('uploader')
        source_type = "Local" if song_data['source'] == 'local_mpd' else "YouTube"
        display_name = title
        if artist and source_type == "Local": display_name = f"{artist} - {title}"
        elif uploader and source_type == "YouTube": display_name = f"{title} (by {uploader})"
        response_lines.append(f"{i+1}. {display_name} [{source_type}]")
    return "\n".join(response_lines)