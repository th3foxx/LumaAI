import asyncio
import json
import logging
import re
import time
from typing import Optional, List, Dict, Any

from settings import settings

logger = logging.getLogger(__name__)

# --- Кэш для результатов YouTube поиска ---
YT_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
YT_CACHE_TTL_SECONDS = settings.music_control.get("youtube_cache_ttl_seconds", 3600) if hasattr(settings, 'music_control') and isinstance(settings.music_control, dict) else 3600

def normalize_youtube_url(url: str) -> Optional[str]:
    """Нормализует URL видео или плейлиста YouTube к каноническому виду."""
    video_id_match = re.search(r"(?:youtu\.be/|youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})", url)
    if video_id_match:
        return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"

    playlist_id_match = re.search(r"youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return f"https://www.youtube.com/playlist?list={playlist_id_match.group(1)}"
    
    return None

async def search_and_get_info(query: str) -> Optional[Dict[str, Any]]:
    """Ищет на YouTube и возвращает информацию о видео или плейлисте."""
    original_query = query
    normalized_url = normalize_youtube_url(query)
    
    query_for_dlp = normalized_url or original_query

    cached_entry = YT_SEARCH_CACHE.get(query_for_dlp)
    if cached_entry and (time.time() - cached_entry["timestamp"]) < YT_CACHE_TTL_SECONDS:
        logger.info(f"Cache hit for YouTube query: '{query_for_dlp}'")
        return cached_entry["data"]
    if cached_entry:
        logger.info(f"Cache expired for YouTube query: '{query_for_dlp}'")
        del YT_SEARCH_CACHE[query_for_dlp]

    logger.info(f"Searching YouTube with yt-dlp for: '{query_for_dlp}'")
    
    yt_dlp_args = [
        'yt-dlp', '--skip-download', '--dump-single-json',
        '--ignore-errors', '--no-warnings', '--format', 'bestaudio[ext=m4a]/bestaudio/best',
    ]
    
    if not normalized_url:
        yt_dlp_args.extend(['--default-search', 'ytsearch1:'])
    
    yt_dlp_args.append(query_for_dlp)

    try:
        process = await asyncio.create_subprocess_exec(
            *yt_dlp_args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)
        stderr_str = stderr_bytes.decode('utf-8', errors='ignore').strip()

        if process.returncode != 0:
            logger.error(f"yt-dlp failed for '{query_for_dlp}' (code {process.returncode}). Stderr: {stderr_str}")
            return None
        
        if not stdout_bytes:
            logger.warning(f"yt-dlp returned no stdout for '{query_for_dlp}'. Stderr: {stderr_str}")
            return None

        result_info = json.loads(stdout_bytes.decode('utf-8', errors='ignore'))
        
        # Внутренняя функция для обработки одной видео-записи
        def _process_video_entry(entry: dict) -> Optional[dict]:
            if not entry or not isinstance(entry, dict) or not entry.get('id'):
                return None
            
            audio_url = entry.get('url')
            if not audio_url and 'formats' in entry:
                audio_formats = [f for f in entry['formats'] if f.get('url') and f.get('acodec') != 'none']
                if audio_formats:
                    # Простая логика выбора лучшего аудио
                    audio_url = sorted(audio_formats, key=lambda x: x.get('abr', 0), reverse=True)[0].get('url')

            if audio_url:
                return {
                    "title": entry.get('title', 'Unknown Title'),
                    "url": audio_url,
                    "uploader": entry.get('uploader', entry.get('channel')),
                    "video_id": entry['id'],
                    "duration": entry.get('duration')
                }
            return None

        # Обработка плейлиста
        if result_info.get("_type") == "playlist" and result_info.get("entries"):
            videos = [v for entry in result_info["entries"] if (v := _process_video_entry(entry)) is not None]
            if not videos: return None
            
            playlist_data = {
                "type": "playlist", "playlist_title": result_info.get("title"),
                "playlist_id": result_info.get("id"), "videos": videos
            }
            YT_SEARCH_CACHE[query_for_dlp] = {"timestamp": time.time(), "data": playlist_data}
            return playlist_data
        
        # Обработка одиночного видео (включая результат поиска)
        else:
            video_data_to_process = result_info.get("entries", [result_info])[0]
            single_video_details = _process_video_entry(video_data_to_process)
            if single_video_details:
                final_data = {"type": "video", **single_video_details}
                YT_SEARCH_CACHE[query_for_dlp] = {"timestamp": time.time(), "data": final_data}
                return final_data
            return None

    except Exception as e:
        logger.error(f"Unexpected error in YouTube search for '{query_for_dlp}': {e}", exc_info=True)
        return None