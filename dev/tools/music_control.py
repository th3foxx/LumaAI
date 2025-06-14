# tools/music_control.py

import asyncio
import logging
import re
import random
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from settings import settings

from utils.mpc_cli import (
    mpc_add_to_playlist, mpc_find_and_add, mpc_get_current_track_details,
    mpc_get_playlist, mpc_get_status, mpc_list_all_tracks, mpc_load_playlist,
    mpc_simple_command, mpc_update_library
)
from utils.youtube import search_and_get_info as search_youtube, normalize_youtube_url

from utils.music_db import (
    add_song_to_liked_db, remove_song_from_liked_db,
    get_all_liked_songs_from_db, is_song_liked_in_db
)

logger = logging.getLogger(__name__)

_CURRENTLY_PLAYING_INFO: Optional[Dict[str, Any]] = None
_CURRENT_YT_PLAYLIST_CONTEXT: Optional[List[Dict[str, Any]]] = None

# --- Вспомогательные функции ---

async def trigger_mpd_library_update() -> bool:
    """Обертка для вызова из main.py"""
    return await mpc_update_library()

async def _update_currently_playing_info_from_mpd() -> bool:
    """Обновляет информацию о текущем треке, используя контекст YouTube, если необходимо."""
    global _CURRENTLY_PLAYING_INFO, _CURRENT_YT_PLAYLIST_CONTEXT
    mpd_details = await mpc_get_current_track_details()

    if not mpd_details or not mpd_details.get("file"):
        _CURRENTLY_PLAYING_INFO = None
        return False

    file_path = mpd_details["file"]
    
    if _CURRENT_YT_PLAYLIST_CONTEXT:
        for track_info in _CURRENT_YT_PLAYLIST_CONTEXT:
            if track_info.get("stream_url") == file_path:
                _CURRENTLY_PLAYING_INFO = {
                    "source": "youtube", "identifier": track_info["video_id"],
                    "title": track_info["title"], "uploader": track_info["uploader"],
                    "url": file_path
                }
                logger.debug(f"Identified YouTube stream via context: {track_info['title']}")
                return True

    if not file_path.startswith("http"):
        _CURRENTLY_PLAYING_INFO = {"source": "local_mpd", "identifier": file_path, **mpd_details}
    else:
        _CURRENTLY_PLAYING_INFO = {"source": "unknown_stream", "identifier": file_path, **mpd_details}
    
    logger.debug(f"Updated playing info: {_CURRENTLY_PLAYING_INFO}")
    return True

def _update_current_playing_info(source: str, identifier: str, **kwargs):
    global _CURRENTLY_PLAYING_INFO
    _CURRENTLY_PLAYING_INFO = {"source": source, "identifier": identifier, **kwargs}
    logger.debug(f"Set currently playing info: {_CURRENTLY_PLAYING_INFO}")

def _clear_youtube_playlist_context():
    global _CURRENT_YT_PLAYLIST_CONTEXT
    if _CURRENT_YT_PLAYLIST_CONTEXT is not None:
        logger.debug("Clearing YouTube playlist context.")
        _CURRENT_YT_PLAYLIST_CONTEXT = None

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
    Plays music from local library or YouTube. Can play specific songs, artists, albums, or playlists.
    If no parameters are given for 'local' source, it resumes or plays random songs.
    """
    global _CURRENT_YT_PLAYLIST_CONTEXT
    logger.info(f"Play music request: song='{song_title}', artist='{artist_name}', query='{search_query}', source='{source}'")

    if source and source.lower() in ["youtube", "internet", "ютуб", "интернет"]:
        query_for_youtube = search_query or " ".join(filter(None, [artist_name, song_title, album_name]))
        if not query_for_youtube: return "Please specify what you want to play from YouTube."
        return await play_from_youtube(query_for_youtube)

    _clear_youtube_playlist_context()
    await mpc_simple_command('clear')

    if playlist_name:
        success, err_load = await mpc_load_playlist(playlist_name)
        if not success: return f"Error loading playlist '{playlist_name}': {err_load}."
        if not await mpc_get_playlist(): return f"Playlist '{playlist_name}' is empty."
        success, err_play = await mpc_simple_command('play')
        if not success: return f"Error starting playback for playlist '{playlist_name}': {err_play}"
        await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
        return f"Playing playlist '{playlist_name}'. Starting with: {_CURRENTLY_PLAYING_INFO.get('title', 'first track')}."

    search_terms_mpc, search_terms_display = [], []
    if artist_name: search_terms_mpc.extend(['artist', artist_name]); search_terms_display.append(f"artist '{artist_name}'")
    if album_name: search_terms_mpc.extend(['album', album_name]); search_terms_display.append(f"album '{album_name}'")
    if song_title: search_terms_mpc.extend(['title', song_title]); search_terms_display.append(f"song '{song_title}'")
    if not search_terms_mpc and search_query: 
        search_terms_mpc.extend(['any', search_query]); search_terms_display.append(f"query '{search_query}'")
    
    if search_terms_mpc:
        success, err_findadd = await mpc_find_and_add(search_terms_mpc)
        if not success: return f"Error finding music for {', '.join(search_terms_display)}: {err_findadd}"
        if not await mpc_get_playlist(): return f"Couldn't find any local music matching: {', '.join(search_terms_display)}."
        success, err_play = await mpc_simple_command('play')
        if not success: return f"Error starting local playback: {err_play}"
        await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
        return f"Playing local music for {', '.join(search_terms_display)}. Now playing: {_CURRENTLY_PLAYING_INFO.get('title', 'first track')}."
    
    else:
        status = await mpc_get_status()
        if status and "[paused]" in status: return await resume_music()
        
        all_tracks = await mpc_list_all_tracks()
        if not all_tracks: return "Your local music library seems empty."
        
        tracks_to_add = random.sample(all_tracks, min(len(all_tracks), random_sample_size))
        for track in tracks_to_add: await mpc_add_to_playlist(track)
        
        if not await mpc_get_playlist(): return "Could not add any random tracks to the playlist."
        await mpc_simple_command('shuffle')
        success, err_play = await mpc_simple_command('play')
        if not success: return f"Error starting random playback: {err_play}"
        await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
        return f"Okay, playing some random music! Now playing: {_CURRENTLY_PLAYING_INFO.get('title', 'a random song')}."

@tool
async def play_from_youtube(search_query: str) -> str:
    """Searches and plays a song, video, or playlist from YouTube."""
    global _CURRENT_YT_PLAYLIST_CONTEXT
    logger.info(f"Attempting to play from YouTube: '{search_query}'")
    if not search_query: return "Please specify a YouTube search query or URL."

    youtube_data = await search_youtube(search_query)
    if not youtube_data: return f"Sorry, I couldn't find anything playable for '{search_query}' on YouTube."

    await mpc_simple_command('clear')
    new_yt_playlist_context = []

    if youtube_data.get("type") == "playlist":
        playlist_title = youtube_data.get("playlist_title", "Unnamed Playlist")
        videos = youtube_data.get("videos", [])
        if not videos: return f"Found playlist '{playlist_title}' but it's empty."

        for video in videos:
            success, err = await mpc_add_to_playlist(video["url"])
            if success: new_yt_playlist_context.append({"video_id": video["video_id"], "title": video["title"], "uploader": video["uploader"], "stream_url": video["url"]})
            else: logger.warning(f"Failed to add '{video['title']}' to playlist: {err}")
        
        if not new_yt_playlist_context: return f"Could not add any videos from playlist '{playlist_title}'."
        _CURRENT_YT_PLAYLIST_CONTEXT = new_yt_playlist_context
        success, err_play = await mpc_simple_command('play')
        if not success: return f"Added {len(new_yt_playlist_context)} videos, but couldn't start playback: {err_play}"
        
        first_video = new_yt_playlist_context[0]
        _update_current_playing_info("youtube", first_video["video_id"], title=first_video["title"], uploader=first_video["uploader"], url=first_video["stream_url"])
        return f"Okay, playing YouTube playlist '{playlist_title}'. Starting with: {first_video['title']}."

    elif youtube_data.get("type") == "video":
        _clear_youtube_playlist_context()
        success, err_add = await mpc_add_to_playlist(youtube_data["url"])
        if not success: return f"Found '{youtube_data['title']}', but couldn't add it to the player: {err_add}."
        
        success, err_play = await mpc_simple_command('play')
        if not success: return f"Added '{youtube_data['title']}', but couldn't start playback: {err_play}"
        
        _CURRENT_YT_PLAYLIST_CONTEXT = [{"video_id": youtube_data["video_id"], "title": youtube_data["title"], "uploader": youtube_data["uploader"], "stream_url": youtube_data["url"]}]
        _update_current_playing_info("youtube", youtube_data["video_id"], title=youtube_data["title"], uploader=youtube_data["uploader"], url=youtube_data["url"])
        return f"Okay, playing '{youtube_data['title']}' by {youtube_data['uploader']} from YouTube."
    
    return "Received unexpected data from YouTube search. Cannot play."

@tool
async def pause_music() -> str:
    """Pauses the currently playing music."""
    status = await mpc_get_status()
    if not status or "[playing]" not in status: return "Nothing is currently playing to pause."
    success, err = await mpc_simple_command('pause')
    return f"Error pausing: {err}" if not success else "Music paused."

@tool
async def resume_music() -> str:
    """Resumes playback of paused music."""
    status = await mpc_get_status()
    if status and "[playing]" in status: return "Music is already playing."
    
    success, err = await mpc_simple_command('play')
    if not success: return f"Error resuming: {err}"
    
    await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
    if _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title", _CURRENTLY_PLAYING_INFO.get("identifier"))
        return f"Resuming playback. Now playing: {title}."
    return "Resuming playback."

@tool
async def stop_music() -> str:
    """Stops the music and clears the playlist."""
    _clear_youtube_playlist_context()
    await mpc_simple_command('stop')
    success, err = await mpc_simple_command('clear')
    _update_current_playing_info(None, None)
    return "Music stopped and playlist cleared." if success else f"Music stopped, but error clearing playlist: {err}"

async def _handle_track_change(command: str) -> str:
    success, err = await mpc_simple_command(command)
    if not success: return f"Error executing '{command}': {err}"
    
    await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
    
    if _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title", _CURRENTLY_PLAYING_INFO.get("identifier"))
        return f"{command.capitalize()}: {title}."
    
    status = await mpc_get_status()
    if status and not ("[playing]" in status or "[paused]" in status):
        _clear_youtube_playlist_context()
        return f"{command.capitalize()}: Reached the end of the playlist."
    return f"{command.capitalize()}: Switched track."

@tool
async def next_song() -> str:
    """Skips to the next song."""
    return await _handle_track_change("next")

@tool
async def previous_song() -> str:
    """Goes back to the previous song."""
    return await _handle_track_change("prev")

@tool
async def set_volume(level: Optional[int] = None, change: Optional[str] = None) -> str:
    """Sets or changes the music player volume."""
    vol_cmd = None
    if level is not None:
        if 0 <= level <= 100: vol_cmd = str(level)
        else: return "Error: Volume level must be between 0 and 100."
    elif change:
        change_lower = change.lower()
        if change_lower.startswith(('+', '-')) and change_lower[1:].isdigit(): vol_cmd = change
        elif change_lower in ['louder', 'громче']: vol_cmd = '+10'
        elif change_lower in ['quieter', 'тише']: vol_cmd = '-10'
        elif change_lower == 'mute': vol_cmd = '0'
        elif change_lower == 'max': vol_cmd = '100'
        else: return f"Error: Unknown volume change command '{change}'."
    else: return "Error: Please specify a volume level or a change amount."

    success, err = await mpc_simple_command('volume', vol_cmd)
    if not success: return f"Error adjusting volume: {err}"
    
    status = await mpc_get_status()
    if status and (match := re.search(r"volume:\s*(\d+)%", status)):
        return f"Volume adjusted. Current volume is {match.group(1)}%."
    return "Volume adjusted."

@tool
async def get_current_song() -> str:
    """Gets information about the currently playing song and player status."""
    await _update_currently_playing_info_from_mpd()
    
    parts = []
    if _CURRENTLY_PLAYING_INFO:
        title = _CURRENTLY_PLAYING_INFO.get("title", _CURRENTLY_PLAYING_INFO.get("identifier"))
        source = _CURRENTLY_PLAYING_INFO.get("source", "unknown").replace('_', ' ').capitalize()
        parts.append(f"Currently: {title} [{source}]")
    else:
        parts.append("Nothing is currently playing.")
    
    status = await mpc_get_status()
    if status:
        if (match := re.search(r"\[(playing|paused|stopped)\]", status)): parts.append(f"Status: {match.group(1)}")
        if (match := re.search(r"volume:\s*(\d+)%", status)): parts.append(f"Volume: {match.group(1)}%")
    else:
        parts.append("Could not get player status.")
    return ". ".join(parts) + "."

@tool
async def like_current_song() -> str:
    """Adds the currently playing song to your liked songs list."""
    if not _CURRENTLY_PLAYING_INFO: await _update_currently_playing_info_from_mpd()
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        return "Nothing seems to be playing, or I can't identify the current song to like it."

    info = _CURRENTLY_PLAYING_INFO
    if info["source"] not in ["local_mpd", "youtube"]: return f"Cannot like song from unsupported source: {info['source']}."
    
    if await is_song_liked_in_db(info["source"], info["identifier"]):
        return f"You've already liked '{info.get('title', info['identifier'])}'."
    
    success = await add_song_to_liked_db(info["source"], info["identifier"], info.get("title"), info.get("artist"), info.get("uploader"))
    return f"Okay, I've added '{info.get('title', info['identifier'])}' to your liked songs." if success else "Sorry, there was a database error."

@tool
async def unlike_current_song() -> str:
    """Removes the currently playing song from your liked songs list."""
    if not _CURRENTLY_PLAYING_INFO: await _update_currently_playing_info_from_mpd()
    if not _CURRENTLY_PLAYING_INFO or not _CURRENTLY_PLAYING_INFO.get("identifier"):
        return "Nothing seems to be playing, or I can't identify the current song to unlike it."

    info = _CURRENTLY_PLAYING_INFO
    if info["source"] not in ["local_mpd", "youtube"]: return f"Cannot unlike song from unsupported source: {info['source']}."
    
    if not await is_song_liked_in_db(info["source"], info["identifier"]):
        return f"It seems '{info.get('title', info['identifier'])}' wasn't in your liked songs list."
    
    success = await remove_song_from_liked_db(info["source"], info["identifier"])
    return f"Okay, I've removed '{info.get('title', info['identifier'])}' from your liked songs." if success else "Sorry, there was a database error."

@tool
async def play_liked_songs(shuffle: Optional[bool] = False) -> str:
    """Plays songs from your liked songs list."""
    global _CURRENT_YT_PLAYLIST_CONTEXT
    liked_songs = await get_all_liked_songs_from_db()
    if not liked_songs: return "You don't have any liked songs yet!"

    await mpc_simple_command('clear')
    new_yt_context = []
    
    # Fetch all YouTube URLs concurrently
    yt_fetch_tasks = [search_youtube(f"https://www.youtube.com/watch?v={s['identifier']}") for s in liked_songs if s['source'] == 'youtube']
    yt_results = await asyncio.gather(*yt_fetch_tasks)
    yt_url_map = {res['video_id']: res['url'] for res in yt_results if res and res.get('url')}

    added_count = 0
    for song in liked_songs:
        url_to_add = None
        if song['source'] == 'local_mpd':
            url_to_add = song['identifier']
        elif song['source'] == 'youtube':
            url_to_add = yt_url_map.get(song['identifier'])
            if url_to_add: new_yt_context.append({"video_id": song["identifier"], "title": song["title"], "uploader": song["uploader"], "stream_url": url_to_add})
        
        if url_to_add:
            success, _ = await mpc_add_to_playlist(url_to_add)
            if success: added_count += 1
            else: logger.warning(f"Failed to add liked song '{song['identifier']}' to MPD.")
    
    if added_count == 0: return "Found liked songs, but couldn't add any to the player."
    
    _CURRENT_YT_PLAYLIST_CONTEXT = new_yt_context if new_yt_context else None
    if shuffle: await mpc_simple_command('shuffle')
    
    success, err = await mpc_simple_command('play')
    if not success: return f"Added {added_count} liked songs, but couldn't start playback: {err}"
    
    await asyncio.sleep(0.2); await _update_currently_playing_info_from_mpd()
    return f"Okay, playing {added_count} liked song(s). Starting with: {_CURRENTLY_PLAYING_INFO.get('title', 'first song')}."

@tool
async def list_liked_songs() -> str:
    """Lists all your liked songs from the database."""
    liked_songs = await get_all_liked_songs_from_db()
    if not liked_songs: return "You haven't liked any songs yet."
    
    lines = ["Here are your liked songs (newest first):"]
    for i, song in enumerate(liked_songs, 1):
        title = song.get('title', song.get('identifier'))
        source = "Local" if song['source'] == 'local_mpd' else "YouTube"
        lines.append(f"{i}. {title} [{source}]")
    return "\n".join(lines)