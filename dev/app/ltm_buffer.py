import asyncio
import logging
from collections import defaultdict
from typing import Optional, Dict, List

from langchain_core.messages import BaseMessage
from mem0 import AsyncMemory

from connectivity import is_internet_available

logger = logging.getLogger(__name__)

# Глобальный буфер и воркер
LTM_MESSAGE_BUFFER: Dict[str, List[BaseMessage]] = defaultdict(list)
LTM_WORKER_TASK: Optional[asyncio.Task] = None
LTM_BUFFER_LOCK = asyncio.Lock()
# Новый параметр: максимальный размер буфера на поток перед принудительной отправкой
MAX_BUFFER_SIZE_PER_THREAD = 30


async def _save_thread_messages_to_ltm(mem0_client: AsyncMemory, thread_id: str, messages: List[BaseMessage]):
    """
    Вспомогательная функция для сохранения сообщений одного потока в LTM.
    Инкапсулирует логику форматирования и вызова mem0.
    """
    if not messages:
        return

    try:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        formatted_messages = [
            {"role": role_map.get(msg.type), "content": msg.content}
            for msg in messages
            if msg.type in role_map and msg.content
        ]

        if not formatted_messages:
            logger.warning(f"LTM Save: No messages to add for thread '{thread_id}' after formatting.")
            return

        logger.info(f"LTM Save: Saving {len(formatted_messages)} messages for thread '{thread_id}'.")
        await mem0_client.add(messages=formatted_messages, user_id=thread_id)
        logger.info(f"LTM Save: Successfully saved context for thread '{thread_id}'.")

    except Exception as e:
        logger.error(f"LTM Save: Failed to save memory for thread '{thread_id}': {e}", exc_info=True)


async def ltm_batch_saving_worker(mem0_client: AsyncMemory, interval_seconds: int):
    """
    Фоновый воркер, который периодически сохраняет сообщения из буфера в mem0.
    Это ЕДИНСТВЕННОЕ место, которое читает из буфера и инициирует сохранение.
    """
    logger.info(f"LTM Batch Saving Worker started. Save interval: {interval_seconds}s.")
    while True:
        try:
            await asyncio.sleep(interval_seconds)

            if not await is_internet_available():
                logger.info("LTM Worker (Timer): Skipping save, no internet connection.")
                continue
            
            buffer_to_process = defaultdict(list)
            async with LTM_BUFFER_LOCK:
                if not LTM_MESSAGE_BUFFER:
                    continue
                
                buffer_to_process = LTM_MESSAGE_BUFFER.copy()
                LTM_MESSAGE_BUFFER.clear()

            logger.info(f"LTM Worker (Timer): Processing buffer with {len(buffer_to_process)} threads.")

            for thread_id, messages_to_add in buffer_to_process.items():
                await _save_thread_messages_to_ltm(mem0_client, thread_id, messages_to_add)

        except asyncio.CancelledError:
            logger.info("LTM Batch Saving Worker is shutting down.")
            if await is_internet_available(force_check=True):
                logger.info("Performing final LTM save on shutdown...")
                async with LTM_BUFFER_LOCK:
                    final_buffer = dict(LTM_MESSAGE_BUFFER)
                    LTM_MESSAGE_BUFFER.clear()
                for thread_id, messages in final_buffer.items():
                    await _save_thread_messages_to_ltm(mem0_client, thread_id, messages)
                logger.info("Final LTM save complete.")
            else:
                logger.warning("Skipping final LTM save on shutdown: no internet connection.")
            break
        except Exception as e:
            logger.error(f"LTM Batch Saving Worker encountered a critical error: {e}", exc_info=True)
            await asyncio.sleep(60)


async def start_ltm_worker(mem0_client: AsyncMemory):
    """Запускает фоновый воркер, если он еще не запущен."""
    global LTM_WORKER_TASK
    if LTM_WORKER_TASK is None or LTM_WORKER_TASK.done():
        if mem0_client:
            save_interval = 300
            LTM_WORKER_TASK = asyncio.create_task(
                ltm_batch_saving_worker(mem0_client, save_interval)
            )
        else:
            logger.warning("Cannot start LTM worker: mem0_client is not available.")


async def stop_ltm_worker():
    """Останавливает фоновый воркер."""
    global LTM_WORKER_TASK
    if LTM_WORKER_TASK and not LTM_WORKER_TASK.done():
        logger.info("Stopping LTM Batch Saving Worker...")
        LTM_WORKER_TASK.cancel()
        try:
            await LTM_WORKER_TASK
        except asyncio.CancelledError:
            pass
        logger.info("LTM Batch Saving Worker stopped.")
    LTM_WORKER_TASK = None