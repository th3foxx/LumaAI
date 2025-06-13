# tools/memory.py

import logging
import asyncio
from typing import Optional, Any, List, Dict
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Эти переменные будут установлены при инициализации в main.py
mem0_client: Optional[Any] = None
mem0_user_id: Optional[str] = None

def initialize_memory_tools(client: Any, user_id: str):
    """
    Инициализирует инструменты памяти, предоставляя им клиент и ID пользователя.
    Это чистый способ внедрения зависимостей без использования глобальных переменных в общем смысле.
    """
    global mem0_client, mem0_user_id
    mem0_client = client
    mem0_user_id = user_id
    logger.info(f"Memory tools initialized for user_id: '{user_id}'")

async def _save_memory_in_background(memory_text: str, user_id: str, metadata: Dict):
    """
    Вспомогательная корутина, которая выполняет реальную работу по сохранению в фоне.
    Она форматирует текст в стандартный для mem0 формат сообщения.
    """
    try:
        # Форматируем одиночный факт в виде диалога из одного сообщения.
        # Это делает вызов консистентным с сохранением контекста из langgraph_llm.py.
        # Добавляем префикс, чтобы mem0 лучше понял, что это явная команда на запоминание.
        formatted_memory = [{
            "role": "user", 
            "content": f"Please remember this fact about me: {memory_text}"
        }]

        # Здесь происходит реальный вызов к mem0, который может быть долгим
        await mem0_client.add(messages=formatted_memory, user_id=user_id, metadata=metadata)
        
        logger.info(f"BACKGROUND SAVE: Successfully tasked adding memory for user '{user_id}': '{memory_text}'")
    except Exception as e:
        logger.error(f"BACKGROUND SAVE FAILED: for user '{user_id}': {e}", exc_info=True)


@tool
async def add_personal_memory(memory_text: str, category: Optional[str] = None) -> str:
    """
    Explicitly adds a piece of text to the long-term memory.
    Useful for when the user says 'remember that...' or 'take a note...'.
    This tool returns immediately and saves the memory in the background.
    """
    if not mem0_client or not mem0_user_id:
        logger.warning("Attempted to use add_personal_memory, but memory client is not initialized.")
        return "Error: The long-term memory service is not available right now."
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    try:
        metadata = {"category": category} if category else {}
        
        # 1. Создаем фоновую задачу для долгой операции сохранения
        asyncio.create_task(
            _save_memory_in_background(memory_text, mem0_user_id, metadata)
        )
        
        logger.info(f"Task to save memory for user '{mem0_user_id}' created. Returning immediately.")
        
        # 2. Немедленно возвращаем ответ, не дожидаясь завершения задачи
        return "OK, I've noted that down and will save it to my long-term memory."

    except Exception as e:
        # Эта ошибка может произойти только если что-то не так с созданием задачи, что маловероятно
        logger.error(f"Failed to CREATE task for adding memory: {e}", exc_info=True)
        return f"Sorry, there was an error trying to initiate the memory saving process: {e}"
    
@tool
async def search_personal_memories(query: str) -> str:
    """
    Searches the long-term memory for information related to the query.
    Use this to recall past conversations, facts about the user, or previously stored notes.
    For example, if the user asks 'What do you know about my preferences?', you should use this tool with a query like 'user preferences'.
    Returns a concise list of the most relevant facts.
    """
    if not mem0_client or not mem0_user_id:
        logger.warning("Attempted to use search_personal_memories, but memory client is not initialized.")
        return "Error: The long-term memory service is not available right now."

    try:
        logger.info(f"Searching memory for user '{mem0_user_id}' with query: '{query}'")
        
        # Запрашиваем топ-5 самых релевантных результатов
        results_data = await mem0_client.search(
            query=query,
            user_id=mem0_user_id,
            limit=5 
        )
        
        memories: List[Dict[str, Any]] = results_data.get("results", [])

        if not memories:
            logger.info("No information found in memory for the query.")
            return "No relevant information found in my long-term memory for this query."

        # Устанавливаем порог релевантности
        SIMILARITY_THRESHOLD = 0.5 # Давайте сделаем его чуть строже
        
        # --- ИСПРАВЛЕННАЯ ЛОГИКА ФИЛЬТРАЦИИ И ЛОГИРОВАНИЯ ---
        
        logger.debug(f"Received {len(memories)} results from mem0. Filtering with threshold > {SIMILARITY_THRESHOLD}")
        
        passed_memories = []
        for item in memories:
            # Пропускаем некорректные или пустые элементы
            if not item or not all(k in item for k in ['memory', 'score']):
                continue
            
            score = item.get('score')
            memory_text = item.get('memory')
            
            logger.debug(f"  - Candidate: '{memory_text}' (Score: {score:.4f})")
            
            if score >= SIMILARITY_THRESHOLD:
                passed_memories.append(memory_text)
                logger.debug(f"    -> PASSED threshold.")
            else:
                logger.debug(f"    -> FAILED threshold.")
        
        # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

        if not passed_memories:
             logger.info(f"Found {len(memories)} memories, but none passed the threshold of {SIMILARITY_THRESHOLD}.")
             return "Found some potentially related information, but nothing with high confidence."

        # Возвращаем отфильтрованные и объединенные результаты
        response = "\n".join(passed_memories)
        
        # Используем правильный формат для лога, чтобы избежать путаницы
        logger.info(f"Returning {len(passed_memories)} relevant memories to LLM: \"{response.replace('\n', ' | ')}\"")
        
        return response

    except Exception as e:
        logger.error(f"Failed to search memory via tool: {e}", exc_info=True)
        return f"Sorry, there was an error trying to search my memory: {e}"