# tools/communication.py
import logging
from typing import Optional, Dict, Tuple, List

import httpx
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
# Это упрощенный подход. В более сложных приложениях управление клиентом может быть вынесено.
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
        # Попытка использовать start_tls=False в конструкторе, если поддерживается
        try:
            smtp_client = aiosmtplib.SMTP(
                hostname=smtp_settings.host,
                port=smtp_settings.port,
                use_tls=is_ssl_connection,
                start_tls=False, # <--- Попытка использовать параметр из примера
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
                raise # Другая ошибка TypeError

        logger.debug("SMTP client object created. Attempting to connect...")
        await smtp_client.connect()
        # Убираем .last_response, так как его нет. Успех connect() уже означает что-то.
        logger.debug(f"Connected to {smtp_client.hostname}:{smtp_client.port}.") 

        if is_tls_connection:
            logger.debug("is_tls_connection is True. Attempting explicit STARTTLS command...")
            try:
                await smtp_client.starttls(tls_context=ssl_context)
            except TypeError:
                logger.warning("aiosmtplib.starttls does not accept tls_context, calling without it.")
                await smtp_client.starttls()
            # Убираем .last_response. Успех starttls() уже означает что-то.
            logger.debug("Explicit STARTTLS command presumed successful.")

        if smtp_settings.username and smtp_settings.password:
            logger.debug(f"Attempting login as {smtp_settings.username}...")
            await smtp_client.login(smtp_settings.username, smtp_settings.password)
            # Убираем .last_response. Успех login() уже означает что-то.
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


async def _send_telegram_message_async(
    chat_id: str,
    contact_name_for_log: str, # Для логирования
    text: str,
    bot_token: str
) -> Tuple[bool, str]:
    """Асинхронно отправляет сообщение в Telegram."""
    if not bot_token:
        logger.error("Telegram Bot Token is not configured.")
        return False, "Ошибка: Токен Telegram бота не настроен в системе."

    # Экранируем специальные символы для MarkdownV2, если планируется его использовать.
    # Для простого текста это не нужно. Если LLM может генерировать Markdown, это важно.
    # Пока отправляем как простой текст (без parse_mode).
    # safe_text = telegram_format_escape(text) # Понадобится функция telegram_format_escape

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        # "parse_mode": "MarkdownV2" # Если используется, текст должен быть экранирован
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("ok"):
                logger.info(f"Telegram message successfully sent to {contact_name_for_log} (chat_id: {chat_id}).")
                return True, f"Сообщение Telegram успешно отправлено контакту {contact_name_for_log}."
            else:
                error_description = response_data.get('description', 'Unknown error')
                logger.error(f"Telegram API error for {contact_name_for_log} (chat_id {chat_id}): {error_description}")
                return False, f"Ошибка API Telegram: {error_description}"
        except httpx.HTTPStatusError as e:
            err_text = e.response.text[:200]
            logger.error(f"HTTP error sending Telegram message to {contact_name_for_log} ({chat_id}): {e.response.status_code} - {err_text}", exc_info=True)
            return False, f"Ошибка HTTP ({e.response.status_code}) при отправке сообщения Telegram."
        except httpx.RequestError as e:
            logger.error(f"Network error sending Telegram message to {contact_name_for_log} ({chat_id}): {e}", exc_info=True)
            return False, "Сетевая ошибка при отправке сообщения Telegram."
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message to {contact_name_for_log} ({chat_id}): {e}", exc_info=True)
            return False, "Непредвиденная ошибка при отправке сообщения Telegram."


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
            # Можно добавить device_model, system_version, app_version для большей "похожести" на реальный клиент
            # device_model="My Voice Assistant",
            # system_version="1.0",
            # app_version="LumiAI"
        )
    
    if not _telethon_client.is_connected():
        try:
            logger.info("Connecting Telethon client...")
            await _telethon_client.connect()
        except Exception as e:
            logger.error(f"Failed to connect Telethon client: {e}", exc_info=True)
            # Важно сбросить клиент, чтобы при следующей попытке он пересоздался
            _telethon_client = None 
            return None

    if not await _telethon_client.is_user_authorized():
        logger.warning("Telethon client is not authorized. Attempting authorization...")
        if not _telethon_client_initialized: # Чтобы не запрашивать телефон постоянно при ошибках
            try:
                # Попытка авторизации. Может потребоваться ввод кода.
                # Если TELEGRAM_CLIENT_PHONE_NUMBER указан, можно использовать его.
                # В противном случае Telethon запросит его в консоли.
                # Это не очень хорошо для фонового сервиса, лучше настроить сессию заранее.
                if settings.telegram.client_phone_number:
                    await _telethon_client.start(phone=settings.telegram.client_phone_number)
                else:
                    # Этот вызов будет ждать ввода в консоль, что плохо для LLM инструмента
                    # Лучше настроить сессию заранее, запустив скрипт с Telethon один раз интерактивно.
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


    _telethon_client_initialized = True # Отмечаем, что инициализация (попытка) была
    return _telethon_client


async def _send_telegram_message_via_client_api(
    contact_identifier: str, # Это может быть username, phone, или численный ID
    contact_name_for_log: str,
    text: str
) -> Tuple[bool, str]:
    """Асинхронно отправляет сообщение через Telegram Client API (от вашего имени)."""
    
    client = await get_telethon_client()
    if not client:
        return False, "Ошибка: Клиент Telegram (для отправки от вашего имени) не инициализирован или не авторизован."

    try:
        # Пытаемся получить сущность получателя (user, chat, channel)
        # Telethon сам разберется, что такое contact_identifier (username, phone, id)
        # Для имен из contacts.json, где telegram_chat_id - это численный ID, это должно работать.
        # Если там username, то тоже.
        try:
            entity = await client.get_entity(contact_identifier)
        except ValueError as e: # Если не удалось распознать идентификатор (например, просто имя "Мама")
            logger.error(f"Could not find Telegram entity for '{contact_identifier}' (contact: {contact_name_for_log}). Error: {e}")
            # Здесь можно добавить логику поиска по имени в контактах, если contact_identifier - это имя
            # Но для contacts.json, где есть telegram_chat_id, это не должно быть проблемой.
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

    except telethon_errors.RPCError as e: # Общая ошибка Telegram API
        logger.error(f"Telegram RPC error sending message to {contact_name_for_log}: {e}", exc_info=True)
        return False, f"Ошибка API Telegram при отправке сообщения: {e}"
    except Exception as e:
        logger.error(f"Unexpected error sending Telegram message via Client API to {contact_name_for_log}: {e}", exc_info=True)
        return False, f"Непредвиденная ошибка при отправке сообщения Telegram от вашего имени."
    # Не отключаем клиент здесь, чтобы он оставался активным для следующих вызовов
    # await client.disconnect() # Отключать лучше при завершении работы приложения


@tool
async def send_message_to_contact(
    contact_name: str,
    message_text: str,
    channel: Optional[str] = None,
    use_personal_telegram: Optional[bool] = False # <--- Новый параметр
) -> str:
    """
    Sends a message to a predefined contact by their name.
    Can use email, Telegram Bot, or your personal Telegram account.
    To send from your personal Telegram account, set 'use_personal_telegram' to True and ensure the channel is 'telegram'.
    If 'use_personal_telegram' is True, the message will appear as if sent directly by you in Telegram.
    Otherwise, Telegram messages are sent via the assistant's bot.
    ...
    Parameters:
        ...
        use_personal_telegram (Optional[bool]): If True and channel is 'telegram' (or chosen automatically as telegram),
                                                 attempts to send from your personal Telegram account. Defaults to False.
                                                 Use this when the user explicitly asks to send a message "from me" or "from my account".
    """
    if not contact_name or not message_text:
        return "Ошибка: Имя контакта и текст сообщения обязательны."

    contact_info = settings.contacts_config.get_contact(contact_name)
    normalized_contact_name_for_user = contact_name.lower()

    if not contact_info:
        # ... (код для отсутствующего контакта, как и раньше)
        available_contacts_msg = ""
        contact_list = settings.contacts_config.available_contacts
        if contact_list:
            available_contacts_msg = f" Доступные контакты: {', '.join(contact_list)}."
        return (f"Ошибка: Контакт '{normalized_contact_name_for_user}' не найден в списке."
                f"{available_contacts_msg} Пожалуйста, используйте одно из предопределенных имен.")

    email_address = contact_info.get("email")
    telegram_chat_id_or_username = contact_info.get("telegram_chat_id") or contact_info.get("telegram_username")
    preferred_channel = contact_info.get("preferred_channel", "").lower()

    chosen_channel = ""
    action_description = ""

    # Определение канала (логика немного усложняется из-за use_personal_telegram)
    if channel:
        channel = channel.lower()
        if channel == "email":
            if email_address: chosen_channel = "email"
            else: return f"Ошибка: У контакта '{normalized_contact_name_for_user}' не настроен email."
        elif channel == "telegram":
            if telegram_chat_id_or_username:
                chosen_channel = "telegram_personal" if use_personal_telegram else "telegram_bot"
            else: return f"Ошибка: У контакта '{normalized_contact_name_for_user}' не настроен Telegram."
        else: return f"Ошибка: Неверный канал '{channel}'. Используйте 'email' или 'telegram'."
    else: # Канал не указан
        # Приоритет: предпочтительный канал, затем Telegram (personal если use_personal_telegram), затем Telegram (bot), затем email
        if preferred_channel == "telegram" and telegram_chat_id_or_username:
            chosen_channel = "telegram_personal" if use_personal_telegram else "telegram_bot"
            action_description = f" (использую предпочтительный канал Telegram{' от вашего имени' if chosen_channel == 'telegram_personal' else ' через бота'})"
        elif preferred_channel == "email" and email_address:
            chosen_channel = "email"
            action_description = " (использую предпочтительный канал email)"
        elif telegram_chat_id_or_username: # Если есть Telegram, но он не предпочтительный
            chosen_channel = "telegram_personal" if use_personal_telegram else "telegram_bot"
            action_description = f" (канал не указан, использую Telegram{' от вашего имени' if chosen_channel == 'telegram_personal' else ' через бота'})"
        elif email_address:
            chosen_channel = "email"
            action_description = " (канал не указан, использую email)"
        else:
            return f"Ошибка: Для контакта '{normalized_contact_name_for_user}' не настроены каналы для отправки."

    # Отправка
    if chosen_channel == "email":
        # ... (код отправки email, как и раньше)
        logger.info(f"Attempting to send email to '{normalized_contact_name_for_user}' ({email_address}){action_description}.")
        email_subject = f"Сообщение от голосового ассистента"
        success, result_message = await _send_email_async(
            recipient_email=email_address,
            recipient_name=normalized_contact_name_for_user.capitalize(),
            subject=email_subject,
            body=message_text,
            smtp_settings=settings.smtp_mail
        )
        return result_message

    elif chosen_channel == "telegram_bot":
        logger.info(f"Attempting to send Telegram message (via Bot) to '{normalized_contact_name_for_user}' (id/username: {telegram_chat_id_or_username}){action_description}.")
        bot_token = settings.telegram.bot_token
        if not bot_token:
             logger.error("Telegram Bot Token is not configured.")
             return "Ошибка: Системная проблема - токен Telegram бота не настроен."
        
        # Убедимся, что telegram_chat_id_or_username это строка для Bot API
        success, result_message = await _send_telegram_message_async(
            chat_id=str(telegram_chat_id_or_username),
            contact_name_for_log=normalized_contact_name_for_user,
            text=message_text,
            bot_token=bot_token
        )
        return result_message

    elif chosen_channel == "telegram_personal":
        logger.info(f"Attempting to send Telegram message (via Client API - personal) to '{normalized_contact_name_for_user}' (id/username: {telegram_chat_id_or_username}){action_description}.")
        
        # Для Telethon, contact_identifier может быть числовым ID или @username
        # Если в contacts.json у вас chat_id, он должен быть числом. Если username, то строкой с @.
        contact_tg_identifier = telegram_chat_id_or_username
        try:
            # Telethon предпочитает числовые ID, если они доступны и это ID пользователя/чата/канала
            contact_tg_identifier = int(telegram_chat_id_or_username)
        except ValueError:
            # Если не число, оставляем как есть (предполагаем, что это username)
            pass

        success, result_message = await _send_telegram_message_via_client_api(
            contact_identifier=contact_tg_identifier, # Передаем ID или username
            contact_name_for_log=normalized_contact_name_for_user,
            text=message_text
        )
        return result_message
    else:
        return f"Критическая ошибка: Не удалось определить канал для '{normalized_contact_name_for_user}'."
    
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
        if contact_info.get("telegram_chat_id"):
            channels.append("Telegram")
        
        # Отображаем имя как оно в файле (если нужно сохранить регистр) или просто name_lower.capitalize()
        # Для простоты, используем name_lower.capitalize()
        display_name = name_lower.capitalize()
        
        if channels:
            response_lines.append(f"- {display_name} (доступно: {', '.join(channels)})")
        else:
            response_lines.append(f"- {display_name} (нет настроенных каналов)")
            
    return "\n".join(response_lines)