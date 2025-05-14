import logging
import httpx # Using httpx for async HTTP requests
from typing import Optional

from settings import settings # To get Telegram token and chat_id

logger = logging.getLogger(__name__)

async def send_telegram_message(message: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Sends a message to a Telegram chat using the HTTP API.
    """
    token = bot_token or settings.telegram.bot_token
    target_chat_id = chat_id or settings.telegram.chat_id

    if not token:
        logger.error("Telegram bot token is not configured. Cannot send message.")
        return False
    if not target_chat_id:
        logger.error("Telegram chat ID is not configured. Cannot send message.")
        return False

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": message,
        "parse_mode": "MarkdownV2" # Or "HTML" or None
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            result = response.json()
            if result.get("ok"):
                logger.info(f"Telegram message sent successfully to chat ID {target_chat_id}.")
                return True
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return False
    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram HTTP error sending message: {e.response.status_code} - {e.response.text}", exc_info=True)
        return False
    except httpx.RequestError as e:
        logger.error(f"Telegram network error sending message: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message: {e}", exc_info=True)
        return False