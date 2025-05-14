import sqlite3
import logging
from datetime import datetime, timezone # Понадобится, если будем работать с liked_at напрямую
from typing import List, Optional, Dict, Any

from settings import settings # Для доступа к settings.music_db_path

logger = logging.getLogger(__name__)

def get_music_db_connection():
    """Establishes a connection to the music SQLite database."""
    conn = sqlite3.connect(settings.music_db_path)
    conn.row_factory = sqlite3.Row 
    return conn

def init_music_likes_table(): # Переименовал для ясности, что это только для этой таблицы
    """Initializes the liked_songs table in the music_data.db if it doesn't exist."""
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_liked_source_id ON liked_songs (source, identifier)")
            conn.commit()
            logger.info(f"Table 'liked_songs' initialized successfully in {settings.music_db_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize 'liked_songs' table in {settings.music_db_path}: {e}", exc_info=True)
        raise 

async def add_song_to_liked_db(source: str, identifier: str, title: Optional[str], artist: Optional[str], uploader: Optional[str]) -> bool:
    """Adds a new liked song to the database or confirms it exists."""
    sql = """
        INSERT OR IGNORE INTO liked_songs (source, identifier, title, artist, uploader)
        VALUES (?, ?, ?, ?, ?) 
    """
    try:
        # Операции с БД лучше выполнять в отдельном потоке, чтобы не блокировать asyncio loop
        # Но для SQLite и редких операций это может быть избыточно.
        # Для простоты оставим синхронный вызов, обернутый в async функцию.
        # В высоконагруженном приложении использовали бы loop.run_in_executor.
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source, identifier, title, artist, uploader))
            conn.commit()
            # lastrowid будет > 0 для новой вставки, total_changes > 0 если была вставка или замена (если бы было REPLACE)
            # Для INSERT OR IGNORE, если запись уже есть, lastrowid может быть не тем, что ожидается,
            # а total_changes будет 0. Проще проверить, есть ли запись после этого.
            # Но для "лайка" достаточно, что она там есть.
            if cursor.lastrowid or conn.total_changes > 0:
                 logger.info(f"Song liked and added/updated in DB: {source} - {identifier} ('{title}')")
                 return True # Успешно добавлено или уже было
            else: # Запись уже существовала, INSERT OR IGNORE ничего не сделал
                 logger.info(f"Song was already liked: {source} - {identifier} ('{title}')")
                 return True # Считаем успехом, если уже есть
    except sqlite3.Error as e:
        logger.error(f"Failed to add liked song to DB: {source} - {identifier} ('{title}'): {e}", exc_info=True)
        return False

async def remove_song_from_liked_db(source: str, identifier: str) -> bool:
    """Removes a song from the liked songs list in the database."""
    sql = "DELETE FROM liked_songs WHERE source = ? AND identifier = ?"
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source, identifier))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Song unliked and removed from DB: {source} - {identifier}")
                return True
            else:
                logger.info(f"Song not found in liked songs or already unliked: {source} - {identifier}")
                return False
    except sqlite3.Error as e:
        logger.error(f"Failed to remove liked song from DB: {source} - {identifier}: {e}", exc_info=True)
        return False

async def get_all_liked_songs_from_db() -> List[Dict[str, Any]]:
    """Retrieves all liked songs from the database, newest first."""
    sql = "SELECT id, source, identifier, title, artist, uploader, liked_at FROM liked_songs ORDER BY liked_at DESC"
    liked_songs = []
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                liked_songs.append(dict(row))
        return liked_songs
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve liked songs: {e}", exc_info=True)
        return []

async def is_song_liked_in_db(source: str, identifier: str) -> bool:
    """Checks if a specific song is already in the liked songs list."""
    sql = "SELECT 1 FROM liked_songs WHERE source = ? AND identifier = ? LIMIT 1"
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source, identifier))
            return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking if song is liked: {source} - {identifier}: {e}", exc_info=True)
        return False # В случае ошибки считаем, что не лайкнут, чтобы избежать проблем