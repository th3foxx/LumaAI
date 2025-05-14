import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

# Imports from your project
from settings import settings
from tools.scheduler import get_due_reminders_and_mark_triggered, DB_PATH # DB_PATH for logging
from utils.notifications import send_telegram_message

# These would be global engine instances injected or accessed from main app context
# For simplicity here, assume they can be imported if main.py sets them up as accessible globals
# A better approach in a larger app would be dependency injection.
# from main import tts_engine, audio_output_engine # This creates a circular dependency if not careful
# Let's assume these are passed to the start_reminder_checker function

logger = logging.getLogger(__name__)

_reminder_checker_task: Optional[asyncio.Task] = None
_stop_event = asyncio.Event()

async def check_and_process_reminders_periodically(tts_engine_ref, audio_output_engine_ref):
    """
    Periodically checks for due reminders and processes them.
    `tts_engine_ref` and `audio_output_engine_ref` are references to the initialized engines.
    """
    logger.info("Reminder checker service started.")
    while not _stop_event.is_set():
        try:
            # logger.debug("Checking for due reminders...")
            due_reminders = await get_due_reminders_and_mark_triggered()

            for reminder in due_reminders:
                reminder_id = reminder.get('id', 'N/A')
                description = reminder.get('description', 'No description')
                original_time = reminder.get('original_time_description', ' unspecified time')
                logger.info(f"Reminder ID {reminder_id} is due: '{description}' (originally set for '{original_time}')")

                # 1. Voice Announcement
                if tts_engine_ref and audio_output_engine_ref and audio_output_engine_ref.is_enabled:
                    full_announcement = f"Напоминаю: {description}"
                    try:
                        logger.info(f"Synthesizing voice announcement for reminder ID {reminder_id}...")
                        # Ensure audio output is running if it was stopped
                        if not audio_output_engine_ref.is_running:
                            logger.info("Audio output engine was not running, starting it for reminder.")
                            audio_output_engine_ref.start()
                        
                        if audio_output_engine_ref.is_running: # Double check
                            tts_audio_bytes = bytearray()
                            async for chunk in tts_engine_ref.synthesize_stream(full_announcement):
                                tts_audio_bytes.extend(chunk)
                            
                            if tts_audio_bytes:
                                logger.info(f"Playing voice announcement for reminder ID {reminder_id} ({len(tts_audio_bytes)} bytes).")
                                audio_output_engine_ref.play_tts_bytes(bytes(tts_audio_bytes))
                            else:
                                logger.warning(f"TTS produced no audio for reminder ID {reminder_id}.")
                        else:
                            logger.warning(f"Audio output engine could not be started for reminder ID {reminder_id}.")

                    except Exception as e:
                        logger.error(f"Error during voice announcement for reminder ID {reminder_id}: {e}", exc_info=True)
                else:
                    logger.warning(f"TTS or Audio Output engine not available/enabled for voice announcement of reminder ID {reminder_id}.")

                # 2. Telegram Notification
                telegram_message = f"🔔 *Напоминание* 🔔\n\n*Что*: {description}\n*Когда было установлено*: {original_time}"
                # Escape MarkdownV2 characters for description and original_time
                # Basic escaping, for more complex cases, a robust library might be needed.
                # Chars to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
                escape_chars = r'_*[]()~`>#+-=|{}.!'
                translator = str.maketrans({char: rf'\{char}' for char in escape_chars})
                
                safe_description = description.translate(translator)
                safe_original_time = original_time.translate(translator)

                telegram_message_md = f"🔔 *Напоминание* 🔔\n\n*Что*: {safe_description}\n*Когда было установлено*: {safe_original_time}"

                logger.info(f"Sending Telegram notification for reminder ID {reminder_id}...")
                await send_telegram_message(telegram_message_md) # Uses MarkdownV2

        except Exception as e:
            logger.error(f"Error in reminder checker loop: {e}", exc_info=True)
        
        try:
            # Wait for the check interval or until stop event is set
            await asyncio.wait_for(_stop_event.wait(), timeout=settings.scheduler_check_interval_seconds)
        except asyncio.TimeoutError:
            continue # Timeout means it's time for the next check
        except Exception as e: # Catch other potential errors from wait_for
            logger.error(f"Error in reminder checker wait: {e}", exc_info=True)
            await asyncio.sleep(settings.scheduler_check_interval_seconds) # Fallback sleep


async def start_reminder_checker(tts_engine_ref, audio_output_engine_ref):
    global _reminder_checker_task, _stop_event
    if _reminder_checker_task is None or _reminder_checker_task.done():
        _stop_event.clear()
        _reminder_checker_task = asyncio.create_task(
            check_and_process_reminders_periodically(tts_engine_ref, audio_output_engine_ref)
        )
        logger.info("Reminder checker background task started.")
    else:
        logger.info("Reminder checker background task already running.")

async def stop_reminder_checker():
    global _reminder_checker_task
    if _reminder_checker_task and not _reminder_checker_task.done():
        logger.info("Stopping reminder checker background task...")
        _stop_event.set() # Signal the loop to stop
        try:
            await asyncio.wait_for(_reminder_checker_task, timeout=5.0) # Wait for it to finish
            logger.info("Reminder checker background task stopped gracefully.")
        except asyncio.TimeoutError:
            logger.warning("Reminder checker background task did not stop gracefully within timeout, cancelling.")
            _reminder_checker_task.cancel()
        except asyncio.CancelledError:
            logger.info("Reminder checker task was cancelled during shutdown.")
        _reminder_checker_task = None
    else:
        logger.info("Reminder checker background task not running or already stopped.")