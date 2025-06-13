import os
import httpx
import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from settings import settings # Убедитесь, что путь к settings правильный

logger = logging.getLogger(__name__)

SERPER_API_KEY = settings.tools.serper_api_key
SERPER_API_URL = settings.tools.serper_api_url

if not SERPER_API_KEY:
    logger.warning(
        "SERPER_API_KEY не установлен в переменных окружения. "
        "Инструмент поиска Serper.dev не будет работать."
    )

@tool
async def search_web_serper(query: str, num_results: Optional[int] = 5, gl: Optional[str] = "ru", hl: Optional[str] = "ru") -> str:
    """
    Searches the web using Serper.dev for a given query and returns a list of search results.
    This tool is useful for finding up-to-date information or answers to general knowledge questions.

    Parameters:
        query (str): The search query.
        num_results (int, optional): The number of search results to return. Defaults to 5. Max is usually 10 for free/basic tiers.
        gl (str, optional): Geolocation. Country code to search from (e.g., 'us', 'ru', 'gb'). Defaults to 'ru'.
        hl (str, optional): Host language. Language interface for the search (e.g., 'en', 'ru'). Defaults to 'ru'.
    """
    if not SERPER_API_KEY:
        logger.error("API ключ для Serper.dev не настроен.")
        return "Ошибка: Сервис поиска Serper.dev не настроен. API ключ отсутствует."
    if not query:
        return "Ошибка: Пожалуйста, укажите поисковый запрос."

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
        "gl": gl,
        "hl": hl
    }

    try:
        logger.info(f"Асинхронный запрос в Serper.dev для '{query}' с параметрами: {payload}")
        async with httpx.AsyncClient(timeout=15.0) as client: # Serper может быть чуть медленнее иногда
            response = await client.post(SERPER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        logger.debug(f"Ответ от API Serper.dev: {data}")

        if not data:
            logger.warning(f"Пустой ответ от Serper.dev для запроса '{query}'")
            return f"Не удалось получить результаты поиска для '{query}'. API вернул пустой ответ."

        # Обработка различных частей ответа Serper
        results_parts = []

        # Answer Box (если есть)
        answer_box = data.get("answerBox")
        if answer_box:
            title = answer_box.get("title", "")
            answer = answer_box.get("answer") or answer_box.get("snippet")
            link = answer_box.get("link")
            if answer:
                results_parts.append(f"Быстрый ответ: {title}\n{answer}{f'\nИсточник: {link}' if link else ''}")

        # Knowledge Graph (если есть)
        knowledge_graph = data.get("knowledgeGraph")
        if knowledge_graph:
            kg_title = knowledge_graph.get("title", "Информация из графа знаний")
            kg_description = knowledge_graph.get("description")
            kg_source_name = knowledge_graph.get("source", {}).get("name")
            kg_source_link = knowledge_graph.get("source", {}).get("link")
            if kg_description:
                results_parts.append(
                    f"{kg_title}:\n{kg_description}"
                    f"{f'\nИсточник: {kg_source_name} ({kg_source_link})' if kg_source_name and kg_source_link else ''}"
                )

        # Organic results
        organic_results = data.get("organic", [])
        if organic_results:
            formatted_organic_results = ["Основные результаты поиска:"]
            for i, result in enumerate(organic_results[:num_results], 1): # Учитываем num_results
                title = result.get("title", "Нет заголовка")
                link = result.get("link", "Нет ссылки")
                snippet = result.get("snippet", "Нет описания").replace("\n", " ") # Убираем переносы строк из сниппета
                formatted_organic_results.append(f"{i}. {title}\n   Ссылка: {link}\n   Описание: {snippet}")
            results_parts.append("\n".join(formatted_organic_results))
        elif not answer_box and not knowledge_graph: # Если нет ничего другого и нет органических результатов
             logger.warning(f"Не найдено результатов поиска (organic) для '{query}' в Serper.dev. Ответ API: {data}")
             return f"По запросу '{query}' ничего не найдено."


        if not results_parts:
            logger.warning(f"Не найдено полезных данных в ответе Serper.dev для '{query}'. Ответ API: {data}")
            return f"По запросу '{query}' не найдено релевантной информации."

        return "\n\n---\n\n".join(results_parts)

    except httpx.HTTPStatusError as e:
        error_message = f"Ошибка при запросе к Serper.dev (HTTP {e.response.status_code})"
        if e.response.status_code == 401 or e.response.status_code == 403: # Unauthorized or Forbidden
            logger.error("Ошибка авторизации с API Serper.dev. Проверьте API ключ.")
            error_message = "Ошибка: Недействительный API ключ или проблема с доступом к Serper.dev."
        elif e.response.status_code == 402: # Payment Required
            logger.error("Закончились кредиты или проблема с оплатой Serper.dev.")
            error_message = "Ошибка: Проблема с биллингом Serper.dev (возможно, закончились бесплатные запросы)."
        elif e.response.status_code == 429: # Too Many Requests
            logger.warning(f"Превышен лимит запросов к Serper.dev для запроса '{query}'.")
            error_message = "Ошибка: Слишком много запросов к сервису поиска. Попробуйте позже."
        else:
            logger.error(f"HTTP ошибка при запросе к Serper.dev для '{query}': {e.response.status_code} - {e.response.text}", exc_info=True)
        return error_message
    except httpx.RequestError as e:
        logger.error(f"Сетевая ошибка при запросе к Serper.dev для '{query}': {e}", exc_info=True)
        return f"Ошибка: Проблема с сетью при попытке выполнить поиск по запросу '{query}'."
    except Exception as e:
        logger.error(f"Неожиданная ошибка при поиске через Serper.dev для '{query}': {e}", exc_info=True)
        return f"Извините, произошла неожиданная ошибка при выполнении поиска по запросу '{query}'."