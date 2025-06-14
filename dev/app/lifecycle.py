"""
Управляет жизненным циклом приложения: инициализация и остановка
всех необходимых движков и сервисов.
"""
import asyncio
import logging
from mem0 import AsyncMemory

# App-specific imports
from .config import settings
from . import globals as G
from .engine_factory import create_engine_instance
from .connection_manager import ConnectionManager

# Project specific imports
from engines.audio_io.input_base import AudioInputEngineBase
from engines.offline_processing.default_processor import DefaultOfflineCommandProcessor
from tools.memory import initialize_memory_tools
from tools.device_control import initialize_device_control_tool

logger = logging.getLogger(__name__)

async def initialize_global_engines():
    """Инициализирует все глобальные движки и сервисы."""
    logger.info("Initializing global engines...")

    G.comm_service = create_engine_instance("communication", settings.engines.communication_engine, settings)
    if G.comm_service:
        await G.comm_service.startup()
        initialize_device_control_tool(G.comm_service)
    else:
        logger.error("Communication service failed to initialize. Dependent services will be affected.")

    if G.comm_service:
        G.offline_command_processor = DefaultOfflineCommandProcessor(comm_service=G.comm_service)
        logger.info("DefaultOfflineCommandProcessor initialized.")
    else:
        G.offline_command_processor = None
        logger.warning("OfflineCommandProcessor not initialized as comm_service is unavailable.")
    
    CUSTOM_FACT_EXTRACTION_PROMPT = """
    You are an expert in extracting personal, long-lasting facts about a user from a conversation.
    Your goal is to identify key pieces of information about the user's identity, preferences, relationships, and plans.
    Extract these as short, concise, third-person facts (e.g., "Likes blue" instead of "I like blue").

    **Guidelines:**
    - The user is the subject of all facts. Pronouns like "I", "me", "my" refer to the user.
    - **CRITICAL: Do NOT extract questions, commands, greetings, or conversational fillers (e.g., 'okay', 'I see'). Only extract new, declarative information about the user.**
    - **CRITICAL: Do NOT extract temporary or one-time information like 'I need to buy milk today' unless it's a long-term plan.**
    - If no personal facts are mentioned, return an empty list of facts.
    - Combine related information from a single user turn into a more descriptive fact.

    **Few-shot examples:**

    Input: Hi, how are you?
    Output: {{"facts": []}}

    Input: I like the color blue.
    Output: {{"facts": ["Likes the color blue"]}}

    Input: My name is Dmitry.
    Output: {{"facts": ["Name is Dmitry"]}}

    Input: I have a friend who lives in Tyumen. His name is Nikita.
    Output: {{"facts": ["Has a friend named Nikita who lives in Tyumen"]}}

    Input: I like pizza, but I'm allergic to mushrooms.
    Output: {{"facts": ["Likes pizza", "Is allergic to mushrooms"]}}

    Input: Remind me to call my mom tomorrow.
    Output: {{"facts": ["Needs to call mom tomorrow"]}} # This is a plan, so it's a valid fact.

    Input: Turn on the light.
    Output: {{"facts": []}} # This is a command, not a fact.

    Return the extracted facts in a JSON format with a "facts" key as shown in the examples.
    """

    CUSTOM_UPDATE_MEMORY_PROMPT = """You are a smart memory manager for a personal AI assistant.
    You can perform four operations on the user's memory: (1) ADD, (2) UPDATE, (3) DELETE, and (4) NONE (no change).

    Compare the "Retrieved facts" with the "Old Memory". For each fact, decide on an operation.

    **Guidelines:**

    1.  **ADD**: If a retrieved fact is completely new and doesn't relate to any existing memory, add it.
        -   Example:
            -   Old Memory: [{"id": "0", "text": "Likes pizza"}]
            -   Retrieved facts: ["Name is Dmitry"]
            -   Result: [{"id": "0", "event": "NONE"}, {"id": "1", "text": "Name is Dmitry", "event": "ADD"}]

    2.  **UPDATE**: If a retrieved fact provides more detail or updates an existing memory, update it.
        -   **CRITICAL RULE:** When updating, you MUST combine the old memory with the new information into a single, more comprehensive fact. **DO NOT lose details from the old memory.**
        -   Example 1 (Good Update):
            -   Old Memory: [{"id": "0", "text": "Has a friend"}]
            -   Retrieved facts: ["Friend's name is Nikita"]
            -   Result: [{"id": "0", "text": "Has a friend named Nikita", "event": "UPDATE", "old_memory": "Has a friend"}]
        -   Example 2 (Good Update - Combining):
            -   Old Memory: [{"id": "0", "text": "Has a friend named Nikita"}]
            -   Retrieved facts: ["Nikita lives in Tyumen"]
            -   Result: [{"id": "0", "text": "Has a friend named Nikita who lives in Tyumen", "event": "UPDATE", "old_memory": "Has a friend named Nikita"}]

    3.  **DELETE**: If a fact is explicitly negated or becomes irrelevant, delete it.
        -   Example:
            -   Old Memory: [{"id": "0", "text": "Likes thriller movies"}]
            -   Retrieved facts: ["Doesn't like thriller movies anymore"]
            -   Result: [{"id": "0", "event": "DELETE"}]

    4.  **NONE**: If a fact is identical or a less-detailed version of an existing memory, do nothing.
        -   Example:
            -   Old Memory: [{"id": "0", "text": "Has a friend named Nikita who lives in Tyumen"}]
            -   Retrieved facts: ["Has a friend in Tyumen"]
            -   Result: [{"id": "0", "event": "NONE"}]

    Your output for the whole memory block must be a single JSON object with a "memory" key.
    """
    CUSTOM_GRAPH_PROMPT = (
        "You are a data extraction expert. From the provided text, which is a single fact about a user, "
        "extract entities and relationships to build a knowledge graph. "
        "The main node is always 'User'. Link all relevant information to it.\n\n"
        "**Rules:**\n"
        "1. Node names (entities) MUST be in TitleCase (e.g., 'Nikita', 'Tyumen', 'Pizza').\n"
        "2. Relationship names MUST be in UPPER_SNAKE_CASE (e.g., 'HAS_FRIEND', 'LIVES_IN').\n"
        "3. Always try to break down complex facts into multiple simple relationships.\n\n"
        "**Examples:**\n\n"
        "Input text: 'Has a friend named Nikita who lives in Tyumen'\n"
        "Output Graph: (User)-[HAS_FRIEND]->(Nikita), (Nikita)-[LIVES_IN]->(Tyumen)\n\n"
        "Input text: 'Likes pizza'\n"
        "Output Graph: (User)-[LIKES]->(Pizza)\n\n"
        "Input text: 'Is allergic to mushrooms'\n"
        "Output Graph: (User)-[IS_ALLERGIC_TO]->(Mushrooms)\n\n"
        "Input text: 'Name is Dmitry'\n"
        "Output Graph: (User)-[HAS_NAME]->(Dmitry)\n\n"
        "Input text: 'Needs to call mom tomorrow'\n"
        "Output Graph: (User)-[HAS_PLAN]->(Call Mom), (Call Mom)-[SCHEDULED_FOR]->(Tomorrow)\n\n"
        "Input text: 'His wife's name is Anna'\n"
        "Output Graph: (User)-[HAS_WIFE]->(Anna)"
    )
    
    # --- Инициализация клиента памяти ---
    if AsyncMemory and settings.mem0.enabled:
        if not all([settings.mem0.neo4j_url, settings.mem0.neo4j_password, settings.mem0.qdrant_url]):
            logger.error("Missing required settings for Mem0. Long-term memory will be disabled.")
        else:
            try:
                mem0_config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "openai_base_url": settings.mem0.openai_api_base,
                            "api_key": settings.mem0.openrouter_api_key,
                            "model": settings.mem0.llm_model,
                            "temperature": 0.2,
                            "max_tokens": 2000,
                        },
                    },
                    "embedder": {
                        "provider": settings.mem0.embedder_provider,
                        "config": {
                            "model": settings.mem0.embedder_model,
                            "embedding_dims": settings.mem0.embedding_dims,
                        },
                    },
                    "graph_store": {
                        "provider": "neo4j",
                        "config": {
                            "url": settings.mem0.neo4j_url,
                            "username": settings.mem0.neo4j_user,
                            "password": settings.mem0.neo4j_password,
                        },
                        "custom_prompt": CUSTOM_GRAPH_PROMPT,
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": settings.mem0.qdrant_collection_name,
                            "url": settings.mem0.qdrant_url,
                            "embedding_model_dims": settings.mem0.embedding_dims,
                        },
                    },
                    "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
                    "custom_update_memory_prompt": CUSTOM_UPDATE_MEMORY_PROMPT,
                    "version": "v1.1",
                }
                
                if settings.mem0.qdrant_api_key:
                    mem0_config["vector_store"]["config"]["api_key"] = settings.mem0.qdrant_api_key

                logger.info(f"Initializing Mem0 client with LLM '{settings.mem0.llm_model}'...")
                G.mem0_client = await AsyncMemory.from_config(mem0_config)
                
                initialize_memory_tools(client=G.mem0_client, user_id=G.ASSISTANT_THREAD_ID)
                logger.info("Mem0 long-term memory client initialized successfully.")

            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client from config: {e}", exc_info=True)
                G.mem0_client = None
    else:
        logger.info("Mem0 is disabled or library not found, skipping long-term memory initialization.")

    G.wake_word_engine = create_engine_instance("wake_word", settings.engines.wake_word_engine, settings)
    G.vad_engine = create_engine_instance("vad", settings.engines.vad_engine, settings)
    G.stt_engine = create_engine_instance("stt", settings.engines.stt_engine, settings)
    G.tts_engine = create_engine_instance("tts", settings.engines.tts_engine, settings)
    G.nlu_engine = create_engine_instance("nlu", settings.engines.nlu_engine, settings)
    G.llm_logic_engine = create_engine_instance("llm_logic", settings.engines.llm_logic_engine, settings)
    G.offline_llm_logic_engine = create_engine_instance("offline_llm_logic", settings.engines.offline_llm_engine, settings)
    
    if not all([G.wake_word_engine, G.vad_engine, G.stt_engine, G.tts_engine, G.nlu_engine, G.comm_service]):
        logger.critical("One or more core processing or communication engines failed to initialize.")
    
    G.manager = ConnectionManager(
        wake_word_engine=G.wake_word_engine,
        vad_engine=G.vad_engine,
        stt_provider=G.stt_engine,
        tts_engine=G.tts_engine,
        nlu_engine=G.nlu_engine,
        llm_logic_engine=G.llm_logic_engine,
        offline_llm_logic_engine=G.offline_llm_logic_engine,
        comm_service=G.comm_service,
        offline_processor=G.offline_command_processor,
        global_audio_settings=settings.audio,
        vad_processing_settings=settings.vad_config,
        sound_device_settings=settings.sounddevice
    )
    logger.info("ConnectionManager initialized.")

    _audio_input_engine_instance = create_engine_instance("audio_input", settings.engines.audio_input_engine, settings)
    if _audio_input_engine_instance and isinstance(_audio_input_engine_instance, AudioInputEngineBase):
        async def process_audio_wrapper(audio_chunk: bytes):
            if G.manager:
                await G.manager.process_audio(audio_chunk, G.ASSISTANT_THREAD_ID, is_local_source=True)
        _audio_input_engine_instance._process_audio_cb = process_audio_wrapper
        G.audio_input_engine = _audio_input_engine_instance
        logger.info(f"AudioInputEngine ({settings.engines.audio_input_engine}) created.")
    else:
        logger.warning(f"Could not create AudioInputEngine: {settings.engines.audio_input_engine}")

    G.audio_output_engine = create_engine_instance("audio_output", settings.engines.audio_output_engine, settings)
    if G.audio_output_engine and G.audio_output_engine.is_enabled:
        logger.info(f"AudioOutputEngine ({settings.engines.audio_output_engine}) created and enabled.")
        if G.manager: G.manager.set_local_audio_output(G.audio_output_engine)
    else:
        logger.info(f"AudioOutputEngine ({settings.engines.audio_output_engine}) is disabled or failed to initialize.")

    if G.tts_engine: await G.tts_engine.startup()
    if G.llm_logic_engine: await G.llm_logic_engine.startup()
    if G.offline_llm_logic_engine: await G.offline_llm_logic_engine.startup()

    logger.info("Global engines initialization phase complete.")


async def shutdown_global_engines():
    """Корректно останавливает все глобальные движки."""
    logger.info("Shutting down global engines...")
    engine_list = [
        G.tts_engine, G.llm_logic_engine, G.offline_llm_logic_engine, G.nlu_engine,
        G.wake_word_engine, G.vad_engine, G.stt_engine, G.audio_input_engine, 
        G.audio_output_engine, G.comm_service
    ]
    for engine in engine_list:
        if engine and hasattr(engine, "shutdown"):
            try:
                logger.info(f"Shutting down {engine.__class__.__name__}...")
                # shutdown can be a coroutine or a regular function
                shutdown_method = engine.shutdown()
                if asyncio.iscoroutine(shutdown_method):
                    await shutdown_method
                logger.info(f"{engine.__class__.__name__} shut down successfully.")
            except Exception as e:
                logger.error(f"Error shutting down {engine.__class__.__name__}: {e}", exc_info=True)
    logger.info("Global engines shutdown phase complete.")