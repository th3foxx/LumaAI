# utils/music_db.py

import aiosqlite
import logging
from typing import List, Optional, Dict, Any

from settings import settings

logger = logging.getLogger(__name__)

def get_music_db_connection() -> aiosqlite.Connection:
    """Returns a coroutine that creates an async connection to the music SQLite database."""
    return aiosqlite.connect(settings.music_db_path)

async def init_music_likes_table():
    """Initializes the liked_songs table in the music_data.db if it doesn't exist."""
    try:
        async with get_music_db_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS liked_songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL CHECK(source IN ('local_mpd', 'youtube')),
                    identifier TEXT NOT NULL, 
                    title TEXT,
                    artist TEXT,
                    uploader TEXT,
                    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unq_liked_song UNIQUE (source, identifier)
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_liked_source_id ON liked_songs (source, identifier)")
            await conn.commit()
            logger.info(f"Table 'liked_songs' initialized successfully in {settings.music_db_path}")
    except aiosqlite.Error as e:
        logger.error(f"Failed to initialize 'liked_songs' table in {settings.music_db_path}: {e}", exc_info=True)
        raise

async def add_song_to_liked_db(source: str, identifier: str, title: Optional[str], 
                               artist: Optional[str], uploader: Optional[str]) -> bool:
    title = title if title and title.strip() else None
    artist = artist if artist and artist.strip() else None
    uploader = uploader if uploader and uploader.strip() else None

    sql = "INSERT OR IGNORE INTO liked_songs (source, identifier, title, artist, uploader) VALUES (?, ?, ?, ?, ?)"
    try:
        async with get_music_db_connection() as conn:
            cursor = await conn.execute(sql, (source, identifier, title, artist, uploader))
            await conn.commit()
            return True # Success if inserted or already exists
    except aiosqlite.Error as e:
        logger.error(f"Failed to add liked song to DB: {source} - {identifier} ('{title}'): {e}", exc_info=True)
        return False

async def remove_song_from_liked_db(source: str, identifier: str) -> bool:
    sql = "DELETE FROM liked_songs WHERE source = ? AND identifier = ?"
    try:
        async with get_music_db_connection() as conn:
            cursor = await conn.execute(sql, (source, identifier))
            await conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Song unliked and removed from DB: {source} - {identifier}")
                return True
            else:
                logger.info(f"Song not found in liked songs for: {source} - {identifier}")
                return False
    except aiosqlite.Error as e:
        logger.error(f"Failed to remove liked song from DB: {source} - {identifier}: {e}", exc_info=True)
        return False

async def get_all_liked_songs_from_db() -> List[Dict[str, Any]]:
    sql = "SELECT id, source, identifier, title, artist, uploader, liked_at FROM liked_songs ORDER BY liked_at DESC"
    liked_songs = []
    try:
        async with get_music_db_connection() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    liked_songs.append(dict(row))
        return liked_songs
    except aiosqlite.Error as e:
        logger.error(f"Failed to retrieve liked songs: {e}", exc_info=True)
        return []

async def is_song_liked_in_db(source: str, identifier: str) -> bool:
    sql = "SELECT 1 FROM liked_songs WHERE source = ? AND identifier = ? LIMIT 1"
    try:
        async with get_music_db_connection() as conn:
            async with conn.execute(sql, (source, identifier)) as cursor:
                return await cursor.fetchone() is not None
    except aiosqlite.Error as e:
        logger.error(f"Error checking if song is liked: {source} - {identifier}: {e}", exc_info=True)
        return False