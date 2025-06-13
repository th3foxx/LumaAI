import sqlite3
import logging
from datetime import datetime, timezone # Понадобится, если будем работать с liked_at напрямую
from typing import List, Optional, Dict, Any

from settings import settings # Для доступа к settings.music_db_path

logger = logging.getLogger(__name__)

def get_music_db_connection():
    """Establishes a connection to the music SQLite database."""
    # Consider enabling WAL mode for better concurrency if needed, though for this use case it might be overkill.
    # conn = sqlite3.connect(settings.music_db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    # conn.execute("PRAGMA journal_mode=WAL;")
    conn = sqlite3.connect(settings.music_db_path)
    conn.row_factory = sqlite3.Row 
    return conn

def init_music_likes_table():
    """Initializes the liked_songs table in the music_data.db if it doesn't exist."""
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            # Ensure source check constraint includes 'youtube'
            # Identifier for 'youtube' will be video_id, for 'local_mpd' it's the file path.
            # Uploader is specific to YouTube, Artist more for local files.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS liked_songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL CHECK(source IN ('local_mpd', 'youtube')),
                    identifier TEXT NOT NULL, 
                    title TEXT,
                    artist TEXT,      -- Primarily for local_mpd source
                    uploader TEXT,    -- Primarily for youtube source
                    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unq_liked_song UNIQUE (source, identifier)
                )
            """)
            # Index for faster lookups, especially for is_song_liked_in_db and deletions
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_liked_source_id ON liked_songs (source, identifier)")
            # Optionally, add an index on liked_at if sorting/filtering by date becomes frequent beyond simple DESC order
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_liked_at ON liked_songs (liked_at)")
            conn.commit()
            logger.info(f"Table 'liked_songs' initialized successfully in {settings.music_db_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize 'liked_songs' table in {settings.music_db_path}: {e}", exc_info=True)
        raise 

async def add_song_to_liked_db(source: str, identifier: str, title: Optional[str], 
                               artist: Optional[str], uploader: Optional[str]) -> bool:
    """Adds a new liked song to the database. Uses INSERT OR IGNORE for idempotency."""
    # Normalize empty strings to None for database consistency
    title = title if title and title.strip() else None
    artist = artist if artist and artist.strip() else None
    uploader = uploader if uploader and uploader.strip() else None

    sql = """
        INSERT OR IGNORE INTO liked_songs (source, identifier, title, artist, uploader)
        VALUES (?, ?, ?, ?, ?) 
    """
    try:
        # For async, ideally use an async DB library or run_in_executor for blocking calls.
        # Sticking to sync for simplicity as per original structure.
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source, identifier, title, artist, uploader))
            conn.commit()
            # INSERT OR IGNORE: rowcount is 1 if inserted, 0 if ignored (already exists).
            # lastrowid is only reliable for actual inserts.
            if cursor.rowcount > 0:
                 logger.info(f"Song liked and added to DB: {source} - {identifier} ('{title}')")
                 return True 
            else:
                 logger.info(f"Song was already liked (or IGNORE prevented insert): {source} - {identifier} ('{title}')")
                 # To be absolutely sure it exists if rowcount is 0, we could do a select,
                 # but for "liking", if it's already there, it's a success.
                 return True # Consider it success if it's already there or was just added.
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
                logger.info(f"Song not found in liked songs (or already unliked) for: {source} - {identifier}")
                return False # Explicitly false if not found to remove
    except sqlite3.Error as e:
        logger.error(f"Failed to remove liked song from DB: {source} - {identifier}: {e}", exc_info=True)
        return False

async def get_all_liked_songs_from_db() -> List[Dict[str, Any]]:
    """Retrieves all liked songs from the database, ordered by when they were liked (newest first)."""
    sql = "SELECT id, source, identifier, title, artist, uploader, liked_at FROM liked_songs ORDER BY liked_at DESC"
    liked_songs = []
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                liked_songs.append(dict(row)) # Convert sqlite3.Row to dict
        return liked_songs
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve liked songs: {e}", exc_info=True)
        return []

async def is_song_liked_in_db(source: str, identifier: str) -> bool:
    """Checks if a specific song (by source and identifier) is already in the liked songs list."""
    sql = "SELECT 1 FROM liked_songs WHERE source = ? AND identifier = ? LIMIT 1"
    try:
        with get_music_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source, identifier))
            return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking if song is liked: {source} - {identifier}: {e}", exc_info=True)
        # In case of error, it's safer to assume not liked to allow re-liking,
        # or handle error upstream. For now, returning False.
        return False

# Call initialization when module is loaded, ensure DB path is set in settings
# This is a side effect on import, consider a dedicated init function called by the application startup.
# if settings.music_db_path:
#    init_music_likes_table()
# else:
#    logger.error("Music DB path not configured in settings. Liked songs functionality will be impaired.")