import aiosqlite
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import dateparser
from settings import settings
from langchain_core.tools import tool

DB_PATH = settings.scheduler_db_path
logger = logging.getLogger(__name__)

# --- Database Setup ---

def get_db_connection() -> aiosqlite.Connection:
    """Returns a coroutine that creates an async connection to the SQLite database."""
    return aiosqlite.connect(DB_PATH)

async def init_db():
    """Initializes the database table if it doesn't exist."""
    try:
        async with get_db_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    due_time_utc TIMESTAMP NOT NULL,
                    original_time_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'triggered', 'cancelled', 'error'))
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_reminders ON reminders (due_time_utc, status)")
            await conn.commit()
            logger.info(f"Database initialized successfully at {DB_PATH}")
    except aiosqlite.Error as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise

# --- Reminder Logic ---

async def add_reminder_to_db(description: str, due_time_utc: datetime, original_time_desc: str) -> Optional[int]:
    sql = "INSERT INTO reminders (description, due_time_utc, original_time_description, status) VALUES (?, ?, ?, 'pending')"
    try:
        async with get_db_connection() as conn:
            cursor = await conn.execute(sql, (description, due_time_utc, original_time_desc))
            await conn.commit()
            reminder_id = cursor.lastrowid
            logger.info(f"Reminder added with ID: {reminder_id}, Due: {due_time_utc.isoformat()}")
            return reminder_id
    except aiosqlite.Error as e:
        logger.error(f"Failed to add reminder to DB: {e}", exc_info=True)
        return None

async def get_pending_reminders_from_db() -> List[Dict[str, Any]]:
    sql = "SELECT id, description, due_time_utc, original_time_description FROM reminders WHERE status = 'pending' ORDER BY due_time_utc ASC"
    reminders = []
    try:
        async with get_db_connection() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    reminders.append(dict(row))
        return reminders
    except aiosqlite.Error as e:
        logger.error(f"Failed to retrieve pending reminders: {e}", exc_info=True)
        return []

async def cancel_reminder_in_db(reminder_id: int) -> bool:
    sql = "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'"
    try:
        async with get_db_connection() as conn:
            cursor = await conn.execute(sql, (reminder_id,))
            await conn.commit()
            return cursor.rowcount > 0
    except aiosqlite.Error as e:
        logger.error(f"Failed to cancel reminder ID {reminder_id}: {e}", exc_info=True)
        return False

async def get_due_reminders_and_mark_triggered() -> List[Dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    select_sql = "SELECT id, description, due_time_utc, original_time_description FROM reminders WHERE due_time_utc <= ? AND status = 'pending'"
    update_sql = "UPDATE reminders SET status = 'triggered' WHERE id = ?"
    due_reminders = []
    try:
        async with get_db_connection() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(select_sql, (now_utc,)) as cursor:
                rows_to_trigger = await cursor.fetchall()

            if not rows_to_trigger:
                return []

            for row in rows_to_trigger:
                reminder_dict = dict(row)
                try:
                    await conn.execute(update_sql, (reminder_dict['id'],))
                    due_reminders.append(reminder_dict)
                except aiosqlite.Error as update_err:
                     logger.error(f"Failed to mark reminder ID {reminder_dict['id']} as triggered: {update_err}", exc_info=True)
            
            await conn.commit()
        return due_reminders
    except aiosqlite.Error as e:
        logger.error(f"Failed to check/trigger due reminders: {e}", exc_info=True)
        return []

# --- Tool Definitions ---

@tool
async def schedule_reminder(description: str, time_description: str) -> str:
    """
    Schedules a reminder for the user.

    Parameters:
        description (str): What the reminder is about (e.g., 'call mom', 'take out the trash').
        time_description (str): When the reminder should trigger (e.g., 'in 10 minutes', 'tomorrow at 3 PM', 'next Tuesday evening', 'at 18:00').
    """
    if not description or not time_description:
        return "Error: Please provide both a description and a time for the reminder."

    logger.info(f"Attempting to schedule reminder: '{description}' at '{time_description}'")

    # Use dateparser to understand the time_description
    # PREFER_DATES_FROM: 'future' helps resolve ambiguities like "5 PM" to today's 5 PM or tomorrow's if it's already past.
    # TODO: Consider making timezone configurable or detecting from user locale if possible
    # For now, assumes time descriptions are relative to the server's local time.
    parsed_time = dateparser.parse(time_description, settings={'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': True})

    if not parsed_time:
        logger.warning(f"Could not parse time description: '{time_description}'")
        # Try to provide better feedback if possible
        common_formats = "'in 5 minutes', 'at 6 PM', 'tomorrow morning', 'next Friday at 10am'"
        return f"Sorry, I couldn't understand the time '{time_description}'. Please try again using formats like {common_formats}."

    # Convert parsed time (which might be local) to UTC for storage
    due_time_utc = parsed_time.astimezone(timezone.utc)
    now_utc = datetime.now(timezone.utc)

    # Sanity check: Don't schedule reminders in the past (allow a small buffer)
    if due_time_utc < (now_utc - timedelta(minutes=1)):
        logger.warning(f"Attempted to schedule reminder in the past: {due_time_utc.isoformat()}")
        return f"Sorry, that time ({parsed_time.strftime('%Y-%m-%d %I:%M %p %Z')}) seems to be in the past. Please provide a future time."

    reminder_id = await add_reminder_to_db(description, due_time_utc, time_description)

    if reminder_id:
        # Format the time nicely for confirmation, using the *parsed* local time before UTC conversion
        # Use %I for 12-hour clock, %p for AM/PM. Adjust format as needed.
        local_due_time = due_time_utc.astimezone() # Convert back to local for display
        time_str = local_due_time.strftime("%A, %B %d at %I:%M %p") # e.g., "Tuesday, July 23 at 05:30 PM"
        return f"Okay, I've scheduled a reminder for '{description}' on {time_str} (ID: {reminder_id})."
    else:
        return "Sorry, there was an error trying to save the reminder. Please try again."


@tool
async def list_reminders() -> str:
    """Lists all currently pending reminders."""
    logger.info("Listing pending reminders.")
    pending = await get_pending_reminders_from_db()

    if not pending:
        return "You have no pending reminders."

    response_lines = ["Here are your pending reminders:"]
    for reminder in pending:
        try:
            # Convert stored UTC time back to local time for display
            due_utc = reminder['due_time_utc']
            # Ensure it's a datetime object if it's stored as a string
            if isinstance(due_utc, str):
                 due_utc = datetime.fromisoformat(due_utc.replace('Z', '+00:00'))
            elif isinstance(due_utc, (int, float)): # Handle timestamps if necessary
                 due_utc = datetime.fromtimestamp(due_utc, timezone.utc)

            # Ensure the datetime object is timezone-aware (UTC)
            if due_utc.tzinfo is None:
                 due_utc = due_utc.replace(tzinfo=timezone.utc)

            local_due_time = due_utc.astimezone() # Convert to local timezone
            time_str = local_due_time.strftime("%Y-%m-%d %I:%M %p %Z") # e.g., 2024-07-23 05:30 PM EDT
            response_lines.append(f"- ID {reminder['id']}: '{reminder['description']}' due around {time_str} (originally: '{reminder['original_time_description']}')")
        except Exception as e:
             logger.error(f"Error formatting reminder {reminder.get('id', 'N/A')}: {e}", exc_info=True)
             response_lines.append(f"- ID {reminder.get('id', 'N/A')}: '{reminder.get('description', 'Error retrieving description')}' (Error formatting time)")


    return "\n".join(response_lines)


@tool
async def cancel_reminder(reminder_id: int) -> str:
    """
    Cancels a pending reminder by its ID.

    Parameters:
        reminder_id (int): The unique ID of the reminder to cancel. Get the ID using list_reminders.
    """
    logger.info(f"Attempting to cancel reminder ID: {reminder_id}")
    if not isinstance(reminder_id, int) or reminder_id <= 0:
        return "Error: Please provide a valid reminder ID (a positive number). You can use 'list reminders' to find the ID."

    success = await cancel_reminder_in_db(reminder_id)
    
    if success:
        return f"Okay, reminder ID {reminder_id} has been cancelled."
    else:
        try:
            async with get_db_connection() as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute("SELECT status FROM reminders WHERE id = ?", (reminder_id,)) as cursor:
                    result = await cursor.fetchone()
            if result:
                return f"Could not cancel reminder ID {reminder_id}. Its current status is '{result['status']}'."
            else:
                return f"Sorry, I couldn't find a reminder with ID {reminder_id}."
        except aiosqlite.Error as e:
             logger.error(f"Database error checking status for reminder ID {reminder_id}: {e}")
             return f"Sorry, I couldn't cancel reminder ID {reminder_id} due to a database error."