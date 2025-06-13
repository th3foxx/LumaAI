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
_CURRENT_YT_PLAYLIST_CONTEXT: Optional[List[Dict[str, Any]]] = None # Stores {video_id, title, uploader, stream_url} for current YT playlist

# --- Кэш для результатов YouTube поиска ---
YT_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
YT_CACHE_TTL_SECONDS = settings.music_control.get("youtube_cache_ttl_seconds", 3600) if hasattr(settings, 'music_control') and isinstance(settings.music_control, dict) else 3600
# Структура YT_SEARCH_CACHE[query] = {"timestamp": time.time(), "data": youtube_info}


# --- Вспомогательные функции для MPC ---

def _run_mpc_command(args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        timeout_duration = 30 if args[0] in ['listall', 'add /', 'update'] else 10
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

async def trigger_mpd_library_update() -> bool:
    logger.info("Attempting to trigger MPD library update (mpc update)...")
    loop = asyncio.get_running_loop()
    stdout, stderr = await loop.run_in_executor(None, _run_mpc_command, ['update'])
    if stderr:
        logger.error(f"Error triggering MPD library update: {stderr}")
        return False
    logger.info(f"MPD library update command sent successfully. MPD is now updating in the background. stdout: {stdout if stdout else 'OK'}")
    return True

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
            title = file_path.split('/')[-1].rsplit('.', 1)[0]
        except:
            title = file_path

    return {"file": file_path, "artist": artist, "title": title, "album": album}

async def _update_currently_playing_info_from_mpd() -> bool:
    global _CURRENTLY_PLAYING_INFO, _CURRENT_YT_PLAYLIST_CONTEXT
    mpd_details = await _get_current_mpd_track_details()

    if not mpd_details or not mpd_details.get("file"):
        _CURRENTLY_PLAYING_INFO = None
        logger.debug("_update_currently_playing_info_from_mpd: No track playing in MPD or no file info.")
        return False

    file_path = mpd_details["file"]
    mpd_artist = mpd_details["artist"]
    mpd_title = mpd_details["title"]
    mpd_album = mpd_details["album"]

    # Check 1: If current MPD track's stream URL matches _CURRENTLY_PLAYING_INFO's URL (already identified)
    if _CURRENTLY_PLAYING_INFO and _CURRENTLY_PLAYING_INFO.get("url") == file_path and _CURRENTLY_PLAYING_INFO.get("source") == "youtube":
        _CURRENTLY_PLAYING_INFO["title"] = mpd_title or _CURRENTLY_PLAYING_INFO.get("title") # Prefer MPD title if available
        logger.debug(f"Re-validated _CURRENTLY_PLAYING_INFO for YouTube stream: {file_path}")
        return True
    if _CURRENTLY_PLAYING_INFO and _CURRENTLY_PLAYING_INFO.get("identifier") == file_path and _CURRENTLY_PLAYING_INFO.get("source") == "local_mpd":
        _CURRENTLY_PLAYING_INFO["title"] = mpd_title or _CURRENTLY_PLAYING_INFO.get("title")
        _CURRENTLY_PLAYING_INFO["artist"] = mpd_artist or _CURRENTLY_PLAYING_INFO.get("artist")
        _CURRENTLY_PLAYING_INFO["album"] = mpd_album or _CURRENTLY_PLAYING_INFO.get("album")
        logger.debug(f"Re-validated _CURRENTLY_PLAYING_INFO for local file: {file_path}")
        return True

    # Check 2: If it's a YouTube stream and we have playlist context
    is_generic_http_stream = file_path.startswith("http") and \
                             ("googlevideo.com/" in file_path or "ytimg.com/" in file_path or ".m3u8" in file_path)

    if is_generic_http_stream and _CURRENT_YT_PLAYLIST_CONTEXT:
        mpd_status_output, _ = _run_mpc_command(['status'])
        current_song_pos_in_mpd_playlist = None
        if mpd_status_output:
            song_match = re.search(r'\[(?:playing|paused)\]\s*#(\d+)/(\d+)', mpd_status_output)
            if song_match:
                current_song_pos_in_mpd_playlist = int(song_match.group(1)) - 1  # 0-indexed

        if current_song_pos_in_mpd_playlist is not None and \
           0 <= current_song_pos_in_mpd_playlist < len(_CURRENT_YT_PLAYLIST_CONTEXT):
            
            track_info_from_context = _CURRENT_YT_PLAYLIST_CONTEXT[current_song_pos_in_mpd_playlist]
            # Verify if the stream URL in context matches the file_path MPD is playing
            if track_info_from_context.get("stream_url") == file_path:
                _CURRENTLY_PLAYING_INFO = {
                    "source": "youtube",
                    "identifier": track_info_from_context["video_id"], 
                    "title": track_info_from_context["title"],
                    "uploader": track_info_from_context["uploader"],
                    "url": file_path # The actual stream URL MPD is playing
                }
                logger.info(f"Identified YouTube stream via playlist context (pos {current_song_pos_in_mpd_playlist}): {track_info_from_context['title']}")
                return True
            else:
                logger.warning(f"MPD playing {file_path}, context stream_url is {track_info_from_context.get('stream_url')} at pos {current_song_pos_in_mpd_playlist}. Mismatch or context outdated.")


    # Check 3: If it's a canonical YouTube URL (youtube.com/watch or youtu.be)
    if "youtube.com/watch?v=" in file_path or "youtu.be/" in file_path:
        logger.info(f"MPD is playing a canonical YouTube URL: {file_path}. Getting video details.")
        youtube_info = await _search_youtube_and_get_info(file_path) 
        if youtube_info and youtube_info.get("type") == "video" and youtube_info.get("video_id"):
            _CURRENTLY_PLAYING_INFO = {
                "source": "youtube",
                "identifier": youtube_info["video_id"], 
                "title": youtube_info.get("title", mpd_title or "YouTube Video"),
                "uploader": youtube_info.get("uploader", mpd_artist),
                "url": youtube_info.get("url") 
            }
            logger.debug(f"Updated playing info for YouTube (from MPD canonical URL): {_CURRENTLY_PLAYING_INFO}")
            return True
        else: 
            _CURRENTLY_PLAYING_INFO = {
                "source": "youtube", "identifier": file_path, 
                "title": mpd_title or "YouTube Stream", "uploader": mpd_artist, "url": file_path
            }
            logger.warning(f"Could not get video_id for YouTube URL from MPD: {file_path}. Using URL as identifier.")
            return True
    
    # Check 4: Local file (not http/https)
    if not (file_path.startswith("http://") or file_path.startswith("https://")):
        _CURRENTLY_PLAYING_INFO = {
            "source": "local_mpd",
            "identifier": file_path,
            "title": mpd_title or file_path.split('/')[-1].rsplit('.', 1)[0],
            "artist": mpd_artist,
            "album": mpd_album
        }
        logger.debug(f"Updated playing info for local_mpd: {_CURRENTLY_PLAYING_INFO}")
        return True

    # Fallback
    logger.warning(f"Could not fully identify track: {file_path}. MPD details: {mpd_details}. Current YT context (first 2 items): {str(_CURRENT_YT_PLAYLIST_CONTEXT[:2]) if _CURRENT_YT_PLAYLIST_CONTEXT else 'None'}")
    _CURRENTLY_PLAYING_INFO = {
        "source": "unknown_stream" if file_path.startswith("http") else "unknown_local",
        "identifier": file_path,
        "title": mpd_title or file_path.split('/')[-1].rsplit('.', 1)[0],
        "artist": mpd_artist, 
        "album": mpd_album,
        "url": file_path if file_path.startswith("http") else None
    }
    if file_path.startswith("http"):
        _CURRENTLY_PLAYING_INFO["uploader"] = mpd_artist 
    return True


def _update_current_playing_info(source: str, identifier: str, title: Optional[str] = None,
                                artist: Optional[str] = None, uploader: Optional[str] = None,
                                album: Optional[str] = None, url: Optional[str] = None):
    global _CURRENTLY_PLAYING_INFO
    _CURRENTLY_PLAYING_INFO = {
        "source": source, "identifier": identifier, "title": title,
        "artist": artist, "uploader": uploader, "album": album, "url": url
    }
    logger.debug(f"Set currently playing info: {_CURRENTLY_PLAYING_INFO}")

def _clear_youtube_playlist_context():
    global _CURRENT_YT_PLAYLIST_CONTEXT
    if _CURRENT_YT_PLAYLIST_CONTEXT is not None:
        logger.debug("Clearing YouTube playlist context.")
        _CURRENT_YT_PLAYLIST_CONTEXT = None

def _normalize_youtube_url(url: str) -> Optional[str]:
    """Normalizes YouTube video or playlist URLs to a canonical form for caching and processing."""
    # Video URL: youtu.be/ID or youtube.com/watch?v=ID
    video_id_match = re.search(r"(?:youtu\.be/|youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})", url)
    if video_id_match:
        return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"

    # Playlist URL: youtube.com/playlist?list=ID
    playlist_id_match = re.search(r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return f"https://www.youtube.com/playlist?list={playlist_id_match.group(1)}"
    
    return None # Not a recognizable YouTube video/playlist URL, treat as search query

async def _search_youtube_and_get_info(query: str) -> Optional[Dict[str, Any]]:
    original_query = query
    normalized_url = _normalize_youtube_url(query)
    
    if normalized_url:
        query_for_cache_and_dlp = normalized_url
        logger.debug(f"Normalized YouTube URL from '{original_query}' to '{query_for_cache_and_dlp}' for search/cache.")
    else:
        query_for_cache_and_dlp = original_query

    cached_entry = YT_SEARCH_CACHE.get(query_for_cache_and_dlp)
    if cached_entry and (time.time() - cached_entry["timestamp"]) < YT_CACHE_TTL_SECONDS:
        logger.info(f"Cache hit for YouTube query: '{query_for_cache_and_dlp}' (Original: '{original_query}')")
        return cached_entry["data"]
    if cached_entry:
        logger.info(f"Cache expired for YouTube query: '{query_for_cache_and_dlp}'")
        del YT_SEARCH_CACHE[query_for_cache_and_dlp]

    logger.info(f"Searching YouTube with yt-dlp for: '{query_for_cache_and_dlp}' (Original: '{original_query}', cache miss or expired)")
    
    yt_dlp_args = [
        'yt-dlp', '--skip-download', '--dump-single-json',
        '--ignore-errors', '--no-warnings', '--format', 'bestaudio[ext=m4a]/bestaudio/best',
    ]
    
    # Add search prefix only if it's not an HTTP/HTTPS URL
    if not (query_for_cache_and_dlp.startswith("http://") or query_for_cache_and_dlp.startswith("https://")):
        yt_dlp_args.extend(['--default-search', 'ytsearch1:'])
    
    yt_dlp_args.append(query_for_cache_and_dlp)

    try:
        process = await asyncio.create_subprocess_exec(
            *yt_dlp_args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)
        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()

        if process.returncode != 0:
            logger.error(f"yt-dlp failed for '{query_for_cache_and_dlp}' (code {process.returncode}). Stderr: {stderr_str}")
            if "Video unavailable" in stderr_str: logger.warning(f"yt-dlp: Video '{query_for_cache_and_dlp}' is unavailable.")
            elif "age restricted" in stderr_str.lower(): logger.warning(f"yt-dlp: Video '{query_for_cache_and_dlp}' is age-restricted.")
            return None
        
        if not stdout_bytes:
            logger.warning(f"yt-dlp returned no stdout for '{query_for_cache_and_dlp}'. Stderr: {stderr_str}"); return None

        result_info = json.loads(stdout_bytes.decode('utf-8', errors='ignore'))

        def process_video_entry(entry_data: Dict[str, Any], original_query_for_fallback_title: str) -> Optional[Dict[str, Any]]:
            if not entry_data or not isinstance(entry_data, dict):
                logger.warning(f"Invalid or empty entry_data in process_video_entry for query '{original_query_for_fallback_title}'")
                return None

            title_from_yt = entry_data.get('title', original_query_for_fallback_title)
            audio_url = entry_data.get('url') 

            if not audio_url and 'formats' in entry_data:
                logger.debug(f"No top-level 'url' in video data for '{title_from_yt}', checking formats list...")
                audio_formats = [f for f in entry_data['formats'] if f.get('url') and f.get('acodec') and f.get('acodec') != 'none']
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
                video_id = entry_data.get('id')
                if not video_id:
                    logger.warning(f"Could not extract video_id for '{title_from_yt}'. Skipping this entry.")
                    return None
                return {
                    "title": title_from_yt, "url": audio_url, 
                    "uploader": entry_data.get('uploader', entry_data.get('channel', "Unknown Uploader")),
                    "video_id": video_id, "duration": entry_data.get('duration')
                }
            else:
                logger.warning(f"Could not extract a playable audio URL for '{title_from_yt}' (ID: {entry_data.get('id')}).")
                return None

        if result_info.get("_type") == "playlist" and "entries" in result_info and result_info["entries"]:
            playlist_videos = []
            for entry in result_info["entries"]:
                if entry is None: 
                    logger.debug("Skipping a null entry in YouTube playlist.")
                    continue
                video_details = process_video_entry(entry, "YouTube Video in Playlist")
                if video_details:
                    playlist_videos.append(video_details)
            
            if not playlist_videos:
                logger.warning(f"YouTube playlist '{query_for_cache_and_dlp}' processed, but no playable videos found/extracted.")
                return None

            playlist_data = {
                "type": "playlist",
                "playlist_title": result_info.get("title", query_for_cache_and_dlp),
                "playlist_id": result_info.get("id"),
                "playlist_uploader": result_info.get("uploader", result_info.get("channel")),
                "videos": playlist_videos
            }
            YT_SEARCH_CACHE[query_for_cache_and_dlp] = {"timestamp": time.time(), "data": playlist_data}
            logger.info(f"Found and cached YouTube playlist: '{query_for_cache_and_dlp}' with {len(playlist_videos)} videos.")
            return playlist_data
        else: 
            video_data_to_process = result_info
            if result_info.get("entries") and isinstance(result_info["entries"], list) and len(result_info["entries"]) > 0:
                 video_data_to_process = result_info["entries"][0]

            if not video_data_to_process or not video_data_to_process.get('id'):
                logger.warning(f"No valid video data to process from yt-dlp output for '{query_for_cache_and_dlp}'. Result was: {str(result_info)[:300]}")
                return None

            single_video_details = process_video_entry(video_data_to_process, query_for_cache_and_dlp)
            if single_video_details:
                final_video_data = {"type": "video", **single_video_details}
                YT_SEARCH_CACHE[query_for_cache_and_dlp] = {"timestamp": time.time(), "data": final_video_data}
                logger.info(f"Found and cached YouTube video: '{final_video_data.get('title')}' for query: '{query_for_cache_and_dlp}'")
                return final_video_data
            else:
                logger.warning(f"Could not get playable details for single video result: '{query_for_cache_and_dlp}' from data: {str(video_data_to_process)[:300]}")
                return None

    except asyncio.TimeoutError: logger.error(f"yt-dlp command timed out for query: {query_for_cache_and_dlp}"); return None
    except json.JSONDecodeError as e: logger.error(f"Failed to parse JSON from yt-dlp for '{query_for_cache_and_dlp}': {e}", exc_info=True); return None
    except FileNotFoundError: logger.error("yt-dlp command not found. Is it installed and in PATH?"); return None
    except Exception as e: logger.error(f"Unexpected error searching YouTube with yt-dlp for '{query_for_cache_and_dlp}': {e}", exc_info=True); return None


# --- Инструменты ---

@tool
async def play_music(
    song_title: Optional[str] = None, artist_name: Optional[str] = None, album_name: Optional[str] = None,
    playlist_name: Optional[str] = None, search_query: Optional[str] = None, source: Optional[str] = "local",
    random_sample_size: int = (
        settings.music_control.get("random_sample_size", 100) 
        if hasattr(settings, 'music_control') and isinstance(settings.music_control, dict) 
        else 100
    )
) -> str:
    """
    Plays music. Can play a specific song, artist, album, or playlist from local library,
    or search and play from YouTube (by setting source to 'youtube' or 'internet').
    Handles YouTube video URLs and YouTube playlist URLs.
    If no specific parameters are given and source is 'local', it attempts to resume playback if paused,
    otherwise it plays random songs from the local library.
    Parameters:
        song_title (Optional[str]): The title of the song to play.
        artist_name (Optional[str]): The name of the artist.
        album_name (Optional[str]): The name of the album.
        playlist_name (Optional[str]): The name of a local MPD playlist to load and play.
        search_query (Optional[str]): A general search query for local files or YouTube (can be a URL).
        source (Optional[str]): 'local' (default) or 'youtube'/'internet' to specify search domain.
        random_sample_size (int): Number of random songs to pick from local library if playing random.
    """
    global _CURRENT_YT_PLAYLIST_CONTEXT
    logger.info(f"Play music request: song='{song_title}', artist='{artist_name}', album='{album_name}', "
                f"playlist='{playlist_name}', query='{search_query}', source='{source}', random_sample_size='{random_sample_size}'")

    if source and source.lower() in ["youtube", "internet", "ютуб", "интернет"]:
        query_for_youtube = search_query
        if not query_for_youtube:
            parts = [part for part in [artist_name, song_title, album_name] if part]
            if not parts: return "Please specify what you want to play from YouTube (e.g., song title, artist, or a YouTube URL)."
            query_for_youtube = " ".join(parts)
        return await play_from_youtube(query_for_youtube)

    _clear_youtube_playlist_context() 
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
                for track_path in tracks_to_add: # This can be slow for large random_sample_size
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
    Searches for a song, video, or playlist on YouTube using the provided query (can be a search term or a direct URL)
    and plays the audio. For playlists, all playable videos are added to the queue.
    Parameters:
        search_query (str): The search term or YouTube URL (video or playlist).
    """
    global _CURRENT_YT_PLAYLIST_CONTEXT
    logger.info(f"Attempting to play from YouTube: '{search_query}'")
    if not search_query: return "Please specify a YouTube search query, video URL, or playlist URL."

    youtube_data = await _search_youtube_and_get_info(search_query)

    if not youtube_data:
        return f"Sorry, I couldn't find anything playable for '{search_query}' on YouTube."

    _run_mpc_command(['clear'])
    
    new_yt_playlist_context = [] 

    if youtube_data.get("type") == "playlist":
        playlist_title = youtube_data.get("playlist_title", "Unnamed Playlist")
        videos_to_play = youtube_data.get("videos", [])
        if not videos_to_play:
            _clear_youtube_playlist_context()
            return f"Found playlist '{playlist_title}' but it seems to be empty or videos are unplayable."

        logger.info(f"Adding YouTube playlist '{playlist_title}' with {len(videos_to_play)} videos to MPD.")
        added_count = 0
        first_video_info_for_update = None

        for video_info in videos_to_play:
            audio_url = video_info.get("url")
            title = video_info.get("title", "YouTube Video")
            video_id = video_info.get("video_id") 
            uploader = video_info.get("uploader")

            if not audio_url or not video_id:
                logger.warning(f"Skipping video '{title}' (ID: {video_id}) from playlist as no stream URL or video_id found.")
                continue

            _, err_add = _run_mpc_command(['add', audio_url])
            if err_add:
                logger.error(f"MPC error adding YouTube URL '{audio_url}' for video '{title}': {err_add}")
            else:
                added_count += 1
                new_yt_playlist_context.append({
                    "video_id": video_id, "title": title, "uploader": uploader, "stream_url": audio_url
                })
                if not first_video_info_for_update:
                    first_video_info_for_update = video_info
        
        if added_count == 0:
            _clear_youtube_playlist_context()
            return f"Could not add any videos from playlist '{playlist_title}' to the player."

        _CURRENT_YT_PLAYLIST_CONTEXT = new_yt_playlist_context 
        _, err_play = _run_mpc_command(['play'])
        if err_play:
            return f"Added {added_count} videos from playlist '{playlist_title}', but couldn't start playback: {err_play}"

        if first_video_info_for_update:
            _update_current_playing_info(
                source="youtube", 
                identifier=first_video_info_for_update["video_id"], 
                title=first_video_info_for_update.get("title"), 
                uploader=first_video_info_for_update.get("uploader"),
                url=first_video_info_for_update.get("url")
            )
        return f"Okay, playing YouTube playlist '{playlist_title}' ({added_count} videos). Starting with: {first_video_info_for_update.get('title', 'first video')}."

    elif youtube_data.get("type") == "video":
        _clear_youtube_playlist_context() 
        audio_url = youtube_data.get("url")
        title = youtube_data.get("title", search_query)
        uploader = youtube_data.get("uploader", "Unknown Artist")
        video_id = youtube_data.get("video_id")

        if not audio_url or not video_id:
            return f"Sorry, I couldn't find a playable stream or video ID for '{title}' on YouTube."

        logger.info(f"Adding YouTube stream to MPD: {title} (ID: {video_id}, URL: {audio_url[:60]}...)")
        _, err_add = _run_mpc_command(['add', audio_url])
        if err_add:
            logger.error(f"MPC error adding YouTube URL '{audio_url}': {err_add}")
            if "Unsupported URI scheme" in err_add or "Invalid argument" in err_add:
                return f"Found '{title}' on YouTube, but the player couldn't handle its stream URL. It might be a protected stream or an unsupported format."
            return f"Found '{title}' on YouTube, but there was an error adding it to the player: {err_add}."
        
        _, err_play = _run_mpc_command(['play'])
        if err_play: return f"Added '{title}' to the player, but couldn't start playback: {err_play}"
        
        # For single video, context is just this one video for immediate identification
        _CURRENT_YT_PLAYLIST_CONTEXT = [{
            "video_id": video_id, "title": title, "uploader": uploader, "stream_url": audio_url
        }]
        _update_current_playing_info(
            source="youtube", identifier=video_id, title=title, 
            uploader=uploader, url=audio_url
        )
        return f"Okay, playing '{title}' by {uploader} from YouTube."
    else:
        _clear_youtube_playlist_context()
        logger.error(f"Received unexpected data structure from YouTube search for '{search_query}': {youtube_data}")
        return f"Received unexpected data structure from YouTube search for '{search_query}'. Cannot play."


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
        playlist_content, _ = _run_mpc_command(['playlist'])
        if not playlist_content:
            return "Nothing to resume. The playlist is empty. Try 'play music'."
    
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
        elif status_after and "[paused]" in status_after: return "Playback is paused (possibly at end of playlist or issue)."
        else: return "Resumed, but it seems the playlist finished, is empty, or player is stopped."


@tool
async def stop_music() -> str:
    """Stops the music playback completely and clears the current playlist."""
    global _CURRENTLY_PLAYING_INFO
    logger.info("Stopping music and clearing playlist.")
    _clear_youtube_playlist_context() 
    _run_mpc_command(['stop']); _, err_clear = _run_mpc_command(['clear'])
    _CURRENTLY_PLAYING_INFO = None
    if err_clear: return f"Music stopped, but error clearing playlist: {err_clear}"
    return "Music stopped and playlist cleared."

async def _handle_track_change(command: str) -> str:
    logger.info(f"Executing MPC command: {command}"); _, err = _run_mpc_command([command])
    if err: return f"Error executing '{command}': {err}"
    
    await asyncio.sleep(0.2); 
    updated = await _update_currently_playing_info_from_mpd()
    
    if updated and _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title")
        artist_or_uploader = _CURRENTLY_PLAYING_INFO.get("artist") or _CURRENTLY_PLAYING_INFO.get("uploader")
        display_name = f"{artist_or_uploader} - {title}" if artist_or_uploader and title else (title or _CURRENTLY_PLAYING_INFO.get("identifier", "next track"))
        return f"{command.capitalize()}: {display_name}."
    else:
        status_after, _ = _run_mpc_command(['status'])
        if status_after and not ("[playing]" in status_after or "[paused]" in status_after):
            _clear_youtube_playlist_context() 
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
    logger.info("Getting current song info."); 
    await _update_currently_playing_info_from_mpd() 
    
    response_parts = []
    if _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title")
        artist = _CURRENTLY_PLAYING_INFO.get("artist"); uploader = _CURRENTLY_PLAYING_INFO.get("uploader")
        album = _CURRENTLY_PLAYING_INFO.get("album"); 
        identifier = _CURRENTLY_PLAYING_INFO.get("identifier", "Unknown track")
        source = _CURRENTLY_PLAYING_INFO.get("source", "unknown source")

        display_title = title or identifier
        if source == "youtube":
            display_title = f"{title}" if title else "YouTube Video"
            if uploader: display_title += f" (by {uploader})"
        elif source == "local_mpd":
            if artist: display_title = f"{artist} - {display_title}"
            if album: display_title += f" (Album: {album})"
        
        response_parts.append(f"Currently: {display_title} [{source.replace('_', ' ').capitalize()}]")
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
    It attempts to identify the song from player status. For YouTube, it uses the video ID.
    For local files, it uses the file path.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        logger.info("Like request: _CURRENTLY_PLAYING_INFO is empty or lacks identifier. Attempting to update.")
        await _update_currently_playing_info_from_mpd()
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to like it."

    source = _CURRENTLY_PLAYING_INFO["source"]
    identifier = _CURRENTLY_PLAYING_INFO["identifier"] 
    title = _CURRENTLY_PLAYING_INFO.get("title")
    artist = _CURRENTLY_PLAYING_INFO.get("artist") 
    uploader = _CURRENTLY_PLAYING_INFO.get("uploader") 

    if source not in ["local_mpd", "youtube"]:
        return f"Cannot like song from unsupported source: {source}. Identifier: {identifier}"
    
    # Ensure YouTube identifier is a video ID, re-fetch if it's a URL (e.g. from unknown_stream fallback)
    if source == "youtube":
        if not re.match(r"^[a-zA-Z0-9_-]{11}$", identifier):
            logger.warning(f"Attempting to like YouTube song with an identifier ('{identifier}') that is not a video ID. Re-checking if it's a URL.")
            normalized_url_for_refetch = _normalize_youtube_url(identifier)
            if normalized_url_for_refetch:
                yt_info = await _search_youtube_and_get_info(normalized_url_for_refetch)
                if yt_info and yt_info.get("type") == "video" and yt_info.get("video_id"):
                    identifier = yt_info["video_id"]
                    title = yt_info.get("title", title)
                    uploader = yt_info.get("uploader", uploader)
                    logger.info(f"Re-identified YouTube song for liking. New ID: {identifier}, Title: {title}")
                else:
                    return f"Could not properly re-identify the YouTube video ID for '{identifier}' to like it."
            else:
                 return f"Cannot like this YouTube item. Identifier '{identifier}' is not a valid video ID or recognizable URL."
        # Update _CURRENTLY_PLAYING_INFO if re-identification occurred
        _CURRENTLY_PLAYING_INFO["identifier"] = identifier
        _CURRENTLY_PLAYING_INFO["title"] = title
        _CURRENTLY_PLAYING_INFO["uploader"] = uploader


    display_name = title or identifier
    if await is_song_liked_in_db(source, identifier): return f"You've already liked '{display_name}'."
    
    success = await add_song_to_liked_db(source, identifier, title, artist, uploader)
    if success: return f"Okay, I've added '{display_name}' to your liked songs."
    else: return f"Sorry, I couldn't add '{display_name}' to your liked songs due to a database error."

@tool
async def unlike_current_song() -> str:
    """
    Removes the currently playing song from your liked songs list.
    Identifies by video ID for YouTube or file path for local songs.
    """
    global _CURRENTLY_PLAYING_INFO
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        logger.info("Unlike request: _CURRENTLY_PLAYING_INFO is empty or lacks identifier. Attempting to update.")
        await _update_currently_playing_info_from_mpd()
        if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
            return "Nothing seems to be playing, or I can't identify the current song to unlike it."

    source = _CURRENTLY_PLAYING_INFO["source"]
    identifier = _CURRENTLY_PLAYING_INFO["identifier"]
    title = _CURRENTLY_PLAYING_INFO.get("title", identifier)

    if source not in ["local_mpd", "youtube"]:
        return f"Cannot unlike song from unsupported source: {source}."

    if source == "youtube":
        if not re.match(r"^[a-zA-Z0-9_-]{11}$", identifier):
            logger.warning(f"Attempting to unlike YouTube song with an identifier ('{identifier}') that is not a video ID. Re-checking if it's a URL.")
            normalized_url_for_refetch = _normalize_youtube_url(identifier)
            if normalized_url_for_refetch:
                yt_info = await _search_youtube_and_get_info(normalized_url_for_refetch)
                if yt_info and yt_info.get("type") == "video" and yt_info.get("video_id"):
                    identifier = yt_info["video_id"]
                    title = yt_info.get("title", title) # Update title for message
                    logger.info(f"Re-identified YouTube song for unliking. New ID: {identifier}, Title: {title}")
                else:
                    return f"Could not properly re-identify the YouTube video ID for '{identifier}' to unlike it."
            else:
                return f"Cannot unlike this YouTube item. Identifier '{identifier}' is not a valid video ID or recognizable URL."
        # Update _CURRENTLY_PLAYING_INFO if re-identification occurred (though not strictly needed for unlike by ID)
        _CURRENTLY_PLAYING_INFO["identifier"] = identifier 
        _CURRENTLY_PLAYING_INFO["title"] = title


    if not await is_song_liked_in_db(source, identifier): return f"It seems '{title}' wasn't in your liked songs list."
    
    success = await remove_song_from_liked_db(source, identifier)
    if success: return f"Okay, I've removed '{title}' from your liked songs."
    else: return f"Sorry, I couldn't remove '{title}' from your liked songs due to a database error."


async def _fetch_youtube_info_for_liked_song(song_data_from_db: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Helper to fetch fresh info for a single liked YouTube song."""
    db_video_id = song_data_from_db['identifier']
    db_title = song_data_from_db.get('title')
    db_uploader = song_data_from_db.get('uploader')

    # Query using the canonical watch URL (which _search_youtube_and_get_info will normalize again, but it's fine)
    youtube_query_url = f"https://www.youtube.com/watch?v={db_video_id}"
    logger.info(f"Fetching fresh info for liked YouTube song (ID: {db_video_id}) via: {youtube_query_url}")
    
    youtube_info = await _search_youtube_and_get_info(youtube_query_url) # Uses cache if available
    
    if youtube_info and youtube_info.get("type") == "video" and youtube_info.get("url") and youtube_info.get("video_id"):
        # Ensure video_id from fresh fetch matches, primarily use fresh details
        if youtube_info["video_id"] != db_video_id:
            logger.warning(f"Video ID mismatch for liked song! DB had {db_video_id}, fetched {youtube_info['video_id']}. Using fetched ID.")
        
        return {
            "source": "youtube",
            "identifier": youtube_info["video_id"], # video_id
            "title": youtube_info.get("title", db_title),
            "uploader": youtube_info.get("uploader", db_uploader),
            "url": youtube_info["url"], # stream URL for MPC
            "original_db_identifier": db_video_id # For logging/debugging if needed
        }
    else: 
        logger.warning(f"Could not get playable URL for liked YouTube song (ID: '{db_video_id}'). Skipping.")
        return None

@tool
async def play_liked_songs(shuffle: Optional[bool] = False) -> str:
    """
    Plays songs from your liked songs list. Clears the current queue and adds all liked songs.
    For YouTube songs, it attempts to fetch fresh stream URLs using their video IDs.
    Parameters:
        shuffle (Optional[bool]): If True, shuffles the liked songs playlist before playing. Defaults to False.
    """
    global _CURRENT_YT_PLAYLIST_CONTEXT
    logger.info(f"Playing liked songs. Shuffle: {shuffle}")
    liked_songs_from_db = await get_all_liked_songs_from_db()
    if not liked_songs_from_db: return "You don't have any liked songs yet!"

    _run_mpc_command(['clear'])
    added_count = 0
    first_track_to_play_info: Optional[Dict[str, Any]] = None
    new_yt_playlist_context_for_liked = []

    youtube_song_fetch_tasks = []
    local_songs_to_process = []

    for song_data in liked_songs_from_db:
        if song_data['source'] == 'youtube':
            # Defer the await, collect tasks
            youtube_song_fetch_tasks.append(_fetch_youtube_info_for_liked_song(song_data))
        elif song_data['source'] == 'local_mpd':
            local_songs_to_process.append({
                "source": "local_mpd",
                "identifier": song_data['identifier'], # file path
                "title": song_data.get('title'),
                "artist": song_data.get('artist'),
                "album": song_data.get('album'),
                "url": song_data['identifier'] # For MPC 'add', path is the URL
            })
        else:
            logger.warning(f"Skipping liked song with unknown source: {song_data['source']} - {song_data['identifier']}")

    processed_youtube_song_infos = []
    if youtube_song_fetch_tasks:
        logger.info(f"Fetching info for {len(youtube_song_fetch_tasks)} liked YouTube songs concurrently...")
        results = await asyncio.gather(*youtube_song_fetch_tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error fetching liked YouTube song info: {res}", exc_info=res)
            elif res: # Successfully fetched and not None
                processed_youtube_song_infos.append(res)
        logger.info(f"Successfully processed {len(processed_youtube_song_infos)} liked YouTube songs.")
    
    all_songs_to_add_to_mpc = local_songs_to_process + processed_youtube_song_infos
    # Note: If shuffle is True, the `first_track_to_play_info` will be the first one *before* MPC shuffles.
    # This is usually acceptable.

    for song_info in all_songs_to_add_to_mpc:
        playable_identifier_for_mpc = song_info['url'] # stream URL for YT, path for local

        _, err = _run_mpc_command(['add', playable_identifier_for_mpc])
        if not err:
            added_count += 1
            if not first_track_to_play_info: 
                first_track_to_play_info = song_info
            
            if song_info['source'] == 'youtube':
                 new_yt_playlist_context_for_liked.append({
                    "video_id": song_info['identifier'], # video_id
                    "title": song_info['title'],
                    "uploader": song_info['uploader'],
                    "stream_url": song_info['url'] # stream_url
                })
        else: 
            original_id_log = song_info.get('original_db_identifier', song_info['identifier'])
            logger.warning(f"Failed to add liked song '{playable_identifier_for_mpc}' (Original ID: {original_id_log}) to MPD: {err}")

    if added_count == 0: 
        _clear_youtube_playlist_context()
        return "Found liked songs, but couldn't add any to the player (possibly due to unavailable YouTube streams or other errors)."

    if new_yt_playlist_context_for_liked:
        _CURRENT_YT_PLAYLIST_CONTEXT = new_yt_playlist_context_for_liked
    else: 
        _clear_youtube_playlist_context()

    if shuffle: _run_mpc_command(['shuffle'])
    
    _, err_play = _run_mpc_command(['play'])
    if err_play: return f"Added {added_count} liked songs, but couldn't start playback: {err_play}"

    if first_track_to_play_info: 
        _update_current_playing_info(
            source=first_track_to_play_info['source'], 
            identifier=first_track_to_play_info['identifier'], 
            title=first_track_to_play_info.get('title'), 
            artist=first_track_to_play_info.get('artist'), 
            uploader=first_track_to_play_info.get('uploader'), 
            album=first_track_to_play_info.get('album'), 
            url=first_track_to_play_info.get('url')
        )
        display_name = first_track_to_play_info.get('title', first_track_to_play_info.get('identifier'))
        return f"Okay, playing {added_count} liked song(s). Starting with: {display_name}."
    else: 
        return f"Okay, playing {added_count} liked song(s)."


@tool
async def list_liked_songs() -> str:
    """
    Lists all your liked songs from the database.
    Displays the title, artist/uploader, and source (Local or YouTube) for each liked song.
    Songs are listed newest first by default (based on when they were liked).
    """
    liked_songs = await get_all_liked_songs_from_db() 
    if not liked_songs: return "You haven't liked any songs yet."
    
    response_lines = ["Here are your liked songs (newest first):"]
    for i, song_data in enumerate(liked_songs):
        title = song_data.get('title', song_data.get('identifier', 'Unknown Title')) 
        artist = song_data.get('artist')    
        uploader = song_data.get('uploader') 
        source_type = "Local" if song_data['source'] == 'local_mpd' else "YouTube"
        
        display_name = title
        if song_data['source'] == 'local_mpd' and artist:
            display_name = f"{artist} - {title}"
        elif song_data['source'] == 'youtube' and uploader:
            display_name = f"{title} (by {uploader})"
        elif song_data['source'] == 'youtube' and not uploader:
             display_name = f"{title}" 

        response_lines.append(f"{i+1}. {display_name} [{source_type}]")
    return "\n".join(response_lines)