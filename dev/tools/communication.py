# tools/communication.py
import logging
from typing import Optional, Dict, Tuple, List

# import httpx # Removed as no longer used for Telegram Bot
import aiosmtplib # Для асинхронной работы с SMTP
from email.mime.text import MIMEText
from email.utils import formataddr # Для корректного отображения имени отправителя

from langchain_core.tools import tool
from settings import settings, SMTPMailSettings # Импортируем SMTPMailSettings для type hinting

from telethon import TelegramClient, errors as telethon_errors # <--- Импорт Telethon
from telethon.tl.types import InputPeerUser, InputPeerChat, InputPeerChannel # Для указания получателя
import ssl # Импортируем ssl для создания контекста


logger = logging.getLogger(__name__)

# Глобальный клиент Telethon, чтобы не создавать его каждый раз
_telethon_client: Optional[TelegramClient] = None
_telethon_client_initialized = False


async def _send_email_async(
    recipient_email: str,
    recipient_name: str,
    subject: str,
    body: str,
    smtp_settings: SMTPMailSettings
) -> Tuple[bool, str]:
    if not all([
        smtp_settings.host,
        smtp_settings.port,
        smtp_settings.username,
        smtp_settings.password,
        smtp_settings.sender_email,
    ]):
        logger.error("SMTP settings are not fully configured.")
        return False, "Ошибка: Настройки SMTP не полностью сконфигурированы."

    msg = MIMEText(body, 'plain', 'utf-8')
    sender_name = "Мой Голосовой Ассистент"
    msg["From"] = formataddr((sender_name, smtp_settings.sender_email))
    msg["To"] = formataddr((recipient_name, recipient_email))
    msg["Subject"] = subject

    is_ssl_connection = False
    is_tls_connection = False

    if smtp_settings.use_tls:
        if smtp_settings.port == 465:
            is_ssl_connection = True
        elif smtp_settings.port == 587:
            is_tls_connection = True
        else:
            is_tls_connection = True # Предполагаем STARTTLS для других портов, если use_tls=True
    
    if is_ssl_connection and is_tls_connection:
        logger.error("SSL (SMTPS) and TLS (STARTTLS) cannot both be effectively True.")
        return False, "Ошибка конфигурации: SSL и TLS не могут быть одновременно True."

    ssl_context = ssl.create_default_context()
    smtp_client = None

    try:
        logger.debug(
            f"Initializing aiosmtplib.SMTP for {smtp_settings.host}:{smtp_settings.port}. "
            f"Calculated: is_ssl_connection={is_ssl_connection}, is_tls_connection={is_tls_connection}."
        )
        try:
            smtp_client = aiosmtplib.SMTP(
                hostname=smtp_settings.host,
                port=smtp_settings.port,
                use_tls=is_ssl_connection,
                start_tls=False, 
                tls_context=ssl_context if is_ssl_connection else None,
                timeout=25
            )
            logger.debug("SMTP client initialized with start_tls=False in constructor.")
        except TypeError as e_init:
            if "unexpected keyword argument 'start_tls'" in str(e_init):
                logger.warning(
                    "Constructor aiosmtplib.SMTP does not support 'start_tls' argument. "
                    "Re-initializing without it."
                )
                smtp_client = aiosmtplib.SMTP(
                    hostname=smtp_settings.host,
                    port=smtp_settings.port,
                    use_tls=is_ssl_connection,
                    tls_context=ssl_context if is_ssl_connection else None,
                    timeout=25
                )
                logger.debug("SMTP client initialized WITHOUT start_tls in constructor.")
            else:
                raise 

        logger.debug("SMTP client object created. Attempting to connect...")
        await smtp_client.connect()
        logger.debug(f"Connected to {smtp_client.hostname}:{smtp_client.port}.") 

        if is_tls_connection:
            logger.debug("is_tls_connection is True. Attempting explicit STARTTLS command...")
            try:
                await smtp_client.starttls(tls_context=ssl_context)
            except TypeError:
                logger.warning("aiosmtplib.starttls does not accept tls_context, calling without it.")
                await smtp_client.starttls()
            logger.debug("Explicit STARTTLS command presumed successful.")

        if smtp_settings.username and smtp_settings.password:
            logger.debug(f"Attempting login as {smtp_settings.username}...")
            await smtp_client.login(smtp_settings.username, smtp_settings.password)
            logger.debug("Login successful.")

        logger.debug("Sending message...")
        await smtp_client.send_message(msg)
        logger.info(f"Email successfully sent to {recipient_name} <{recipient_email}> with subject: '{subject}'")
        
        return True, f"Email успешно отправлен контакту {recipient_name}."

    except aiosmtplib.SMTPConnectError as e:
        logger.error(f"SMTPConnectError for {recipient_name}: {e}", exc_info=True)
        return False, f"Ошибка подключения к SMTP серверу ({getattr(e, 'code', 'N/A')}): {getattr(e, 'message', str(e))}"
    except aiosmtplib.SMTPHeloError as e:
        logger.error(f"SMTPHeloError for {recipient_name}: {e}", exc_info=True)
        return False, f"Ошибка SMTP HELO/EHLO ({getattr(e, 'code', 'N/A')}): {getattr(e, 'message', str(e))}"
    except aiosmtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTPAuthenticationError for {recipient_name}: {e}", exc_info=True)
        return False, f"Ошибка аутентификации SMTP ({getattr(e, 'code', 'N/A')}): {getattr(e, 'message', str(e))}. Проверьте логин и пароль."
    except aiosmtplib.SMTPResponseException as e:
        logger.error(f"SMTPResponseException for {recipient_name}: {e}", exc_info=True)
        if "Connection already using TLS" in getattr(e, 'message', str(e)) and is_tls_connection:
             logger.critical("CRITICAL: Got 'Connection already using TLS' on port 587. This means 'start_tls=False' in constructor did not prevent auto-TLS, or explicit starttls() was still problematic.")
        return False, f"Ошибка ответа SMTP сервера ({getattr(e, 'code', 'N/A')}): {getattr(e, 'message', str(e))}"
    except aiosmtplib.SMTPException as e:
        logger.error(f"SMTPException for {recipient_name}: {e}", exc_info=True)
        return False, f"Общая ошибка SMTP: {str(e)}"
    except ssl.SSLError as e:
        logger.error(f"SSLError during SMTP for {recipient_name}: {e}", exc_info=True)
        return False, f"Ошибка SSL при отправке email: {str(e)}"
    except ConnectionRefusedError as e:
        logger.error(f"ConnectionRefusedError for {recipient_name}: {e}", exc_info=True)
        return False, f"Не удалось подключиться к SMTP серверу (соединение отклонено)."
    except Exception as e:
        logger.error(f"Unexpected error for {recipient_name}: {e}", exc_info=True)
        return False, f"Непредвиденная ошибка при отправке email: {str(e)}"
    finally:
        if smtp_client and smtp_client.is_connected:
            try:
                await smtp_client.quit()
                logger.debug("SMTP client quit and disconnected.")
            except Exception as e_quit:
                logger.error(f"Error during SMTP quit: {e_quit}", exc_info=True)

# Removed _send_telegram_message_async as Telegram Bot sending is no longer used.
# httpx import was also removed.

async def get_telethon_client() -> Optional[TelegramClient]:
    global _telethon_client, _telethon_client_initialized

    if not settings.telegram.client_api_id or not settings.telegram.client_api_hash:
        logger.error("Telegram Client API ID или API Hash не настроены.")
        return None

    if _telethon_client is None:
        logger.info(f"Initializing Telethon client with session: {settings.telegram.client_session_name}")
        _telethon_client = TelegramClient(
            settings.telegram.client_session_name,
            settings.telegram.client_api_id,
            settings.telegram.client_api_hash,
        )
    
    if not _telethon_client.is_connected():
        try:
            logger.info("Connecting Telethon client...")
            await _telethon_client.connect()
        except Exception as e:
            logger.error(f"Failed to connect Telethon client: {e}", exc_info=True)
            _telethon_client = None 
            return None

    if not await _telethon_client.is_user_authorized():
        logger.warning("Telethon client is not authorized. Attempting authorization...")
        if not _telethon_client_initialized: 
            try:
                if settings.telegram.client_phone_number:
                    await _telethon_client.start(phone=settings.telegram.client_phone_number)
                else:
                    logger.error("TELEGRAM_CLIENT_PHONE_NUMBER не указан, интерактивная авторизация невозможна в этом контексте.")
                    await _telethon_client.disconnect()
                    _telethon_client = None
                    return None
                logger.info("Telethon client authorized successfully.")
                _telethon_client_initialized = True
            except Exception as e:
                logger.error(f"Telethon client authorization failed: {e}", exc_info=True)
                if _telethon_client and _telethon_client.is_connected():
                    await _telethon_client.disconnect()
                _telethon_client = None
                return None
        else:
            logger.error("Telethon client authorization failed previously, not retrying automatically.")
            if _telethon_client and _telethon_client.is_connected():
                await _telethon_client.disconnect()
            _telethon_client = None
            return None

    _telethon_client_initialized = True 
    return _telethon_client


async def _send_telegram_message_via_client_api(
    contact_identifier: str, 
    contact_name_for_log: str,
    text: str
) -> Tuple[bool, str]:
    """Асинхронно отправляет сообщение через Telegram Client API (от вашего имени)."""
    
    client = await get_telethon_client()
    if not client:
        return False, "Ошибка: Клиент Telegram (для отправки от вашего имени) не инициализирован или не авторизован."

    try:
        try:
            entity = await client.get_entity(contact_identifier)
        except ValueError as e: 
            logger.error(f"Could not find Telegram entity for '{contact_identifier}' (contact: {contact_name_for_log}). Error: {e}")
            return False, f"Не удалось найти получателя '{contact_name_for_log}' в Telegram по идентификатору '{contact_identifier}'. Убедитесь, что идентификатор (username или ID) корректен."
        except telethon_errors.FloodWaitError as e:
            logger.warning(f"Flood wait error from Telegram: {e.seconds} seconds. Cannot send message now.")
            return False, f"Telegram просит подождать {e.seconds} секунд перед отправкой следующего сообщения (flood control)."
        except Exception as e:
            logger.error(f"Error getting entity for '{contact_identifier}' (contact: {contact_name_for_log}): {e}", exc_info=True)
            return False, f"Ошибка при поиске получателя '{contact_name_for_log}' в Telegram."

        logger.info(f"Sending message via Telethon to {contact_name_for_log} (entity: {entity.id}). Text: {text[:30]}...")
        await client.send_message(entity, text)
        logger.info(f"Telegram message successfully sent via Client API to {contact_name_for_log}.")
        return True, f"Сообщение Telegram успешно отправлено (от вашего имени) контакту {contact_name_for_log}."

    except telethon_errors.RPCError as e: 
        logger.error(f"Telegram RPC error sending message to {contact_name_for_log}: {e}", exc_info=True)
        return False, f"Ошибка API Telegram при отправке сообщения: {e}"
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message via Client API to {contact_name_for_log}: {e}", exc_info=True)
        return False, f"Непредвиденная ошибка при отправке сообщения Telegram от вашего имени."

@tool
async def send_email_to_contact(
    contact_name: str,
    subject: str,
    message_text: str
) -> str:
    """
    Sends an email to a predefined contact by their name.

    Parameters:
        contact_name (str): The name of the contact as defined in the contacts configuration.
        subject (str): The subject of the email.
        message_text (str): The body content of the email.
    """
    if not contact_name or not message_text or not subject:
        return "Ошибка: Имя контакта, тема и текст сообщения обязательны для отправки email."

    contact_info = settings.contacts_config.get_contact(contact_name)
    normalized_contact_name_for_user = contact_name.lower() # For consistency in messages

    if not contact_info:
        available_contacts_msg = ""
        contact_list = settings.contacts_config.available_contacts
        if contact_list:
            available_contacts_msg = f" Доступные контакты: {', '.join(contact_list)}."
        return (f"Ошибка: Контакт '{normalized_contact_name_for_user}' не найден в списке."
                f"{available_contacts_msg} Пожалуйста, используйте одно из предопределенных имен.")

    email_address = contact_info.get("email")

    if not email_address:
        return f"Ошибка: У контакта '{normalized_contact_name_for_user}' не настроен email адрес."

    logger.info(f"Attempting to send email to '{normalized_contact_name_for_user}' ({email_address}).")
    success, result_message = await _send_email_async(
        recipient_email=email_address,
        recipient_name=contact_name.capitalize(), # Use original or capitalized for display
        subject=subject,
        body=message_text,
        smtp_settings=settings.smtp_mail
    )
    return result_message

@tool
async def send_telegram_message_to_contact(
    contact_name: str,
    message_text: str
) -> str:
    """
    Sends a Telegram message from your personal account to a predefined contact by their name.
    This requires your personal Telegram account to be configured and authorized via Telethon.

    Parameters:
        contact_name (str): The name of the contact as defined in the contacts configuration.
                               This contact must have a 'telegram_chat_id' or 'telegram_username'.
        message_text (str): The content of the message to send.
    """
    if not contact_name or not message_text:
        return "Ошибка: Имя контакта и текст сообщения обязательны для отправки Telegram сообщения."

    contact_info = settings.contacts_config.get_contact(contact_name)
    normalized_contact_name_for_user = contact_name.lower() # For consistency in messages

    if not contact_info:
        available_contacts_msg = ""
        contact_list = settings.contacts_config.available_contacts
        if contact_list:
            available_contacts_msg = f" Доступные контакты: {', '.join(contact_list)}."
        return (f"Ошибка: Контакт '{normalized_contact_name_for_user}' не найден в списке."
                f"{available_contacts_msg} Пожалуйста, используйте одно из предопределенных имен.")

    telegram_identifier = contact_info.get("telegram_chat_id") or contact_info.get("telegram_username")

    if not telegram_identifier:
        return f"Ошибка: У контакта '{normalized_contact_name_for_user}' не настроен идентификатор Telegram (chat_id или username)."

    logger.info(f"Attempting to send Telegram message (via Client API - personal) to '{normalized_contact_name_for_user}' (id/username: {telegram_identifier}).")
    
    contact_tg_identifier_for_api = telegram_identifier
    try:
        # Telethon prefers numerical IDs if available and it's a user/chat/channel ID
        contact_tg_identifier_for_api = int(telegram_identifier)
    except ValueError:
        # If not a number, assume it's a username (string)
        pass

    success, result_message = await _send_telegram_message_via_client_api(
        contact_identifier=contact_tg_identifier_for_api,
        contact_name_for_log=contact_name, # Use original or capitalized for logging
        text=message_text
    )
    return result_message
    
@tool
def list_contacts() -> str:
    """
    Lists all available contacts to whom messages can be sent.
    """
    contact_list = settings.contacts_config.available_contacts
    if not contact_list:
        return "У вас пока нет сохраненных контактов."
    
    response_lines = ["Вот ваши сохраненные контакты:"]
    for name_lower in contact_list:
        contact_info = settings.contacts_config.get_contact(name_lower)
        channels = []
        if contact_info.get("email"):
            channels.append("email")
        if contact_info.get("telegram_chat_id") or contact_info.get("telegram_username"):
            # Now "Telegram" implies sending from personal account
            channels.append("Telegram (личный)") 
        
        display_name = name_lower.capitalize()
        
        if channels:
            response_lines.append(f"- {display_name} (доступно: {', '.join(channels)})")
        else:
            response_lines.append(f"- {display_name} (нет настроенных каналов)")
            
    return "\n".join(response_lines)