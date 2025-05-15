import logging
import httpx
from typing import Optional, Dict, Any, List # Добавлен List
from urllib.parse import quote

from langchain_core.tools import tool
from settings import settings

logger = logging.getLogger(__name__)


def _format_jina_response(query: str, response_json: Dict[str, Any]) -> str:
    """
    Форматирует ответ от Jina Search API (s.jina.ai) в строку, подходящую для голосового ассистента.
    Ожидаемая структура ответа: {"data": [{"title": "...", "content": "...", "url": "..."}, ...]}
    """
    results: Optional[List[Dict[str, Any]]] = response_json.get("data")

    if not results or not isinstance(results, list) or not results: # Проверяем, что results это непустой список
        logger.warning(f"Jina Search для '{query}' не вернул результатов в поле 'data' или формат неожиданный: {response_json}")
        return f"К сожалению, по запросу '{query}' мне не удалось найти конкретную информацию через Jina Search, или ответ был в неожиданном формате."

    first_result = results[0]
    title = first_result.get('title', 'Найден результат').strip()
    content_snippet = first_result.get('content', '').strip()
    # url = first_result.get('url') # URL пока не используем в голосовом ответе

    if not content_snippet and not title:
        return f"По вашему запросу '{query}' Jina Search вернул пустой результат."
    
    if not content_snippet:
        if title != 'Найден результат':
            return f"По вашему запросу '{query}', Jina Search нашел: {title}. К сожалению, подробное описание отсутствует."
        else: # title дефолтный и нет контента
            return f"По вашему запросу '{query}', Jina Search нашел результат, но без подробного описания."

    # Сокращаем фрагмент для голосового ответа
    max_snippet_len_voice = 220  # Можно настроить
    processed_snippet = content_snippet

    if len(content_snippet) > max_snippet_len_voice:
        content_snippet_trimmed = content_snippet[:max_snippet_len_voice]
        # Пытаемся обрезать по предложению
        last_period = content_snippet_trimmed.rfind('.')
        last_question = content_snippet_trimmed.rfind('?')
        last_exclamation = content_snippet_trimmed.rfind('!')
        last_sentence_end = max(last_period, last_question, last_exclamation)

        if last_sentence_end > max_snippet_len_voice * 0.4: # Убедимся, что обрезка не слишком короткая
             processed_snippet = content_snippet[:last_sentence_end+1].strip()
        else:
             # Обрезаем по слову, если предложение слишком короткое или не найдено
             last_space = content_snippet_trimmed.rfind(' ')
             if last_space != -1:
                 processed_snippet = content_snippet_trimmed[:last_space].strip() + "..."
             else: # Если нет пробелов (одно длинное слово)
                 processed_snippet = content_snippet_trimmed.strip() + "..."
    
    # Формируем ответ
    if title and title.lower() != "найден результат" and title.lower() not in processed_snippet.lower()[:len(title)+15]:
        # Если заголовок информативен и не является началом контента, добавляем его
        response_str = f"По вашему запросу '{query}', Jina Search сообщает: {title}. {processed_snippet}"
    else:
        # Если заголовок не очень полезен или уже есть в контенте
        response_str = f"По вашему запросу '{query}', Jina Search нашел: {processed_snippet}"

    # Убираем лишние пробелы и повторы
    response_str = " ".join(response_str.split())

    if len(results) > 1:
        response_str += " Также есть и другие результаты по этому запросу."
        
    return response_str


@tool
async def search_web_jina(query: str) -> str:
    """
    Searches the web using Jina Search to find up-to-date information, news, or answers to specific questions.
    Use this tool when you need current information that might not be in your training data,
    such as recent events, specific facts, or detailed explanations.

    Parameters:
        query (str): The search query or question (e.g., 'latest news on AI advancements', 'what is the current CEO of OpenAI?', 'explain how black holes are formed').
    """
    if not query:
        logger.warning("Попытка поиска с пустым запросом.")
        return "Ошибка: Пожалуйста, укажите поисковый запрос."

    try:
        api_url_base = settings.tools.jina_search_api_url
        api_key = settings.tools.jina_search_api_key
    except AttributeError as e:
        logger.error(f"Ошибка доступа к настройкам Jina Search (jina_search_api_url или jina_search_api_key): {e}")
        return "Ошибка конфигурации: не удалось получить настройки для Jina Search."

    if not api_key:
        logger.error(
            "API ключ Jina Search (JINA_SEARCH_API_KEY) не настроен в settings.tools.jina_search_api_key. "
            "Этот ключ необходим для использования Jina Search."
        )
        return (
            "Ошибка: API ключ для Jina Search не настроен. "
            "Пожалуйста, получите ключ на jina.ai и добавьте его в конфигурацию (переменная окружения JINA_SEARCH_API_KEY)."
        )

    encoded_query = quote(query)
    full_url = f"{api_url_base.rstrip('/')}/{encoded_query}"
    
    params = {"json": "true"} 
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}" 
        # Примечание: если "Bearer <key>" не сработает, Jina может использовать "token <key>" или просто ключ в "X-Api-Key".
        # Судя по общей практике, "Bearer" наиболее вероятен для новых API.
    }

    logger.info(f"Выполняется Jina Search по запросу: '{query}' URL: {full_url}")
    logger.debug(f"Используется API ключ Jina Search (начинается с: {api_key[:4]}...).")

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(full_url, params=params, headers=headers)
            response.raise_for_status() 
            data = response.json()
        
        logger.debug(f"Ответ от Jina Search API для '{query}': {data}")
        return _format_jina_response(query, data)

    except httpx.HTTPStatusError as e:
        error_body = ""
        try:
            error_body = e.response.json() # Попробуем получить JSON из тела ошибки
        except Exception:
            error_body = e.response.text[:200] # Или просто текст
        
        logger.error(
            f"Ошибка API Jina Search для '{query}': {e.response.status_code} - {error_body}",
            exc_info=True # Печатаем полный traceback в лог для отладки
        )
        if e.response.status_code == 401:
             return "Ошибка: Аутентификация с Jina Search не удалась. Пожалуйста, проверьте ваш API ключ (JINA_SEARCH_API_KEY)."
        elif e.response.status_code == 403: # Forbidden
             return "Ошибка: Доступ к Jina Search запрещен. Возможно, ваш API ключ не имеет необходимых прав или истек срок действия."
        elif e.response.status_code == 404: 
             return f"Ошибка: Jina Search не смог найти информацию по запросу '{query}' (ресурс не найден)."
        elif e.response.status_code == 429: # Too Many Requests
             return "Ошибка: Слишком много запросов к Jina Search. Пожалуйста, попробуйте позже."
        return f"Ошибка: Jina Search вернул ошибку со статусом {e.response.status_code}. Не удалось получить информацию по запросу '{query}'."
    except httpx.RequestError as e:
        logger.error(f"Сетевая ошибка при выполнении Jina Search для '{query}': {e}", exc_info=True)
        return f"Ошибка: Проблема с сетью при попытке обращения к Jina Search по запросу '{query}'."
    except Exception as e:
        logger.error(f"Неожиданная ошибка при выполнении Jina Search для '{query}': {e}", exc_info=True)
        return f"К сожалению, произошла непредвиденная ошибка при поиске информации по запросу '{query}'."