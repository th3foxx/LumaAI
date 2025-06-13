# --- START OF FILE engines/llm_logic/langgraph_llm.py (Async _call_model) ---
import asyncio
import atexit
import logging
from functools import cache
from typing import Union, Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver 

from tools import TOOLS # TOOLS should contain async tools where appropriate

from .base import LLMLogicEngineBase
from settings import AISettings, PostgresSettings

logger = logging.getLogger(__name__)

def init_embeddings_internal(model_name: str, ai_settings: AISettings):
    if "openai" in model_name.lower() or "gpt" in model_name.lower():
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=ai_settings.openai_api_key,
            openai_api_base=ai_settings.openai_api_base if ai_settings.openai_api_base else None
        )
    elif "google_vertexai" in model_name.lower():
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
            return VertexAIEmbeddings(model_name=model_name.split(':')[-1].strip())
        except ImportError:
            logger.error("langchain-google-vertexai not installed. Cannot use VertexAI embeddings.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize VertexAIEmbeddings: {e}")
            raise
    else:
        raise ValueError(f"Unsupported embedding model type in init_embeddings_internal: {model_name}")


class LangGraphLLMEngine(LLMLogicEngineBase):
    def __init__(self, config: Dict[str, Any]):
        self.ai_settings: AISettings = config.get("ai_settings")
        self.postgres_settings: PostgresSettings = config.get("postgres_settings")
        
        if not all([self.ai_settings, self.postgres_settings]):
            raise ValueError("LangGraphLLMEngine requires 'ai_settings' and 'postgres_settings' in config")

        self._llm: Optional[ChatOpenAI] = None
        self._agent_model: Optional[Any] = None
        self._tool_node: Optional[ToolNode] = None
        self._store_cm = None
        self._store: Optional[BaseStore] = None
        self._checkpointer = MemorySaver() 
        self._app: Optional[Any] = None
        self._system_prompt: Optional[SystemMessage] = None
        self._atexit_registered = False

    def _get_store(self) -> BaseStore:
        if self._store is None:
            logger.info(f"LLM Engine: Initializing PostgresStore for vector search with URI: {self.postgres_settings.uri.split('@')[-1]}")
            try:
                embeddings = init_embeddings_internal(self.ai_settings.embedding_model, self.ai_settings)
            except Exception as e:
                logger.error(f"LLM Engine: Failed to initialize embedding model '{self.ai_settings.embedding_model}': {e}", exc_info=True)
                raise RuntimeError(f"Could not initialize embedding model for PostgresStore: {e}") from e

            index_cfg = {"dims": self.ai_settings.embedding_dims, "embed": embeddings}
            try:
                self._store_cm = PostgresStore.from_conn_string(self.postgres_settings.uri, index=index_cfg)
                self._store = self._store_cm.__enter__()
                self._store.setup()
                logger.info("LLM Engine: PostgresStore (for vector search) initialized and setup complete.")
                if not self._atexit_registered:
                    atexit.register(self._close_store_atexit)
                    self._atexit_registered = True
            except Exception as e:
                logger.error(f"LLM Engine: Failed to initialize or setup PostgresStore: {e}", exc_info=True)
                raise RuntimeError("Could not connect to or setup the LangGraph PostgresStore.") from e
        return self._store

    def _close_store_atexit(self):
        if self._store_cm:
            logger.info("LLM Engine (atexit): Closing PostgresStore connection...")
            try:
                self._store_cm.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"LLM Engine (atexit): Error closing PostgresStore: {e}", exc_info=True)
            self._store = None; self._store_cm = None

    def _filter_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        if not messages: return []
        has_system = isinstance(messages[0], SystemMessage)
        system = messages[0] if has_system else None
        history_start_index = 1 if has_system else 0
        recent = messages[max(history_start_index, len(messages) - self.ai_settings.history_length):]
        filtered = []
        if system: filtered.append(system)
        filtered.extend(recent)
        return filtered

    async def _call_model(self, state: MessagesState) -> MessagesState: # <<< MADE ASYNC
        filtered_msgs = self._filter_messages(state["messages"])
        if not filtered_msgs:
            logger.warning("LLM Engine: _call_model received empty state messages after filtering.")
            return {"messages": [HumanMessage(content="Error: No messages to process.")]}
        
        logger.debug(f"LLM Engine: Calling agent model with {len(filtered_msgs)} messages.")
        try:
            response = await self._agent_model.ainvoke(filtered_msgs) # <<< CHANGED TO AINVOKE
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM Engine: Error invoking agent model: {e}", exc_info=True)
            error_msg = HumanMessage(content=f"Sorry, I encountered an error with the AI model: {e}")
            return {"messages": [error_msg]}

    def _should_continue(self, state: MessagesState) -> str:
        # This node is typically CPU-bound (inspecting state), so sync is fine.
        last_msg = state["messages"][-1] if state["messages"] else None
        if last_msg and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            if not any(tc.get("name") for tc in last_msg.tool_calls):
                logger.warning("LLM Engine: Agent returned tool_calls, but they are empty. Ending turn.")
                return END
            logger.info(f"LLM Engine: Agent requested tool(s): {[tc.get('name') for tc in last_msg.tool_calls if tc.get('name')]}") # Added safety check
            return "action"
        else:
            logger.info("LLM Engine: Agent did not request tools. Ending turn.")
            return END

    def _build_workflow(self) -> StateGraph:
        # Workflow definition itself doesn't change whether nodes are sync or async
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model) # LangGraph handles async node
        workflow.add_node("action", self._tool_node) 
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"action": "action", END: END}
        )
        workflow.add_edge("action", "agent")
        logger.info("LLM Engine: LangGraph workflow built.")
        return workflow

    @cache
    def _get_app_and_prompt_internal(self) -> tuple[Any, SystemMessage]:
        logger.info("LLM Engine: Initializing LangGraph app and system prompt...")
        
        try:
            self._llm = ChatOpenAI(
                openai_api_base=self.ai_settings.openai_api_base if self.ai_settings.openai_api_base else None,
                openai_api_key=self.ai_settings.openai_api_key,
                model_name=self.ai_settings.grok_model,
                temperature=self.ai_settings.temperature,
            )
            # .with_structured_output can also be used if tools require complex inputs
            self._agent_model = self._llm.bind_tools(TOOLS) 
            self._tool_node = ToolNode(TOOLS) 
            logger.info(f"LLM Engine: ChatOpenAI initialized with model: {self.ai_settings.grok_model}")
        except Exception as e:
            logger.error(f"LLM Engine: Failed to initialize ChatOpenAI: {e}", exc_info=True)
            raise RuntimeError("Could not initialize the LLM for the engine.") from e

        store_instance = self._get_store()
        workflow = self._build_workflow()
        
        app = workflow.compile(
            checkpointer=self._checkpointer, 
            store=store_instance            
        )
        logger.info("LLM Engine: LangGraph app compiled.")

        system_prompt = SystemMessage(
            content=(
                "You are Lumi (female gender), an efficient AI voice assistant. Your primary goal is to fulfill the user's request quickly and accurately. "
                "Always respond in Russian unless asked otherwise."
                "First, analyze the user's request. "
                "DECIDE QUICKLY: Can you answer directly based on the conversation history or your general knowledge? "
                "If YES, provide a concise answer immediately without using tools. "
                "If NO, and the request requires external action (like controlling a device, searching memory, etc.) or information you don't have, IDENTIFY the necessary tool and its arguments. "
                "**When you decide to use a tool, your response should consist *only* of the tool call(s). Do not add any conversational text or explanation before making the tool call.** " 
                "After the tool runs, use its output to formulate your final, concise response to the user."
                "Keep your spoken responses natural and brief."
            )
        )
        logger.info("LLM Engine: System prompt created.")
        return app, system_prompt

    async def startup(self):
        try:
            self._get_store() 
            self._app, self._system_prompt = self._get_app_and_prompt_internal()
            logger.info("LangGraphLLMEngine started and app compiled.")
        except Exception as e:
            logger.error(f"LangGraphLLMEngine startup failed: {e}", exc_info=True)
            raise

    async def shutdown(self):
        if self._store_cm:
            logger.info("LLM Engine (shutdown): Closing PostgresStore connection...")
            try:
                self._store_cm.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"LLM Engine (shutdown): Error closing PostgresStore: {e}", exc_info=True)
            self._store = None; self._store_cm = None
            logger.info("LLM Engine (shutdown): PostgresStore connection closed.")
        
        if hasattr(self._get_app_and_prompt_internal, 'cache_clear'):
            self._get_app_and_prompt_internal.cache_clear()
        
        self._app = None; self._system_prompt = None
        self._llm = None; self._agent_model = None; self._tool_node = None
        
        if hasattr(self._checkpointer, 'close') and callable(self._checkpointer.close):
            try:
                logger.info("LLM Engine (shutdown): Closing checkpointer...")
                if asyncio.iscoroutinefunction(self._checkpointer.close): await self._checkpointer.close()
                else: self._checkpointer.close()
                logger.info("LLM Engine (shutdown): Checkpointer closed.")
            except Exception as e:
                logger.error(f"LLM Engine (shutdown): Error closing checkpointer: {e}", exc_info=True)
        logger.info("LangGraphLLMEngine resources cleared/closed.")

    async def ask(self, question: str, thread_id: str, return_full: bool = False) -> Union[str, BaseMessage]:
        if not self._app or not self._system_prompt:
            logger.error("LLM Engine: App/System Prompt not initialized. Call startup() first.")
            error_content = "Sorry, the language model is not available right now."
            return AIMessage(content=error_content) if return_full else error_content

        logger.info(f"LLM Engine: ask called for thread '{thread_id}'. Question: '{question[:50]}...'") # Renamed from ask_lumi
        try:
            messages = [self._system_prompt, HumanMessage(content=question)]
            config = {"configurable": {"thread_id": thread_id}}
            final_event = None
            async for event in self._app.astream( {"messages": messages}, config=config, stream_mode="values" ):
                final_event = event

            if final_event is None or not final_event.get("messages"):
                logger.error(f"LLM Engine: Agent did not return any events/messages for thread '{thread_id}'.")
                raise RuntimeError("Lumi (LLM Engine) did not return a valid response.")

            final_msg = final_event["messages"][-1]
            logger.info(f"LLM Engine: Lumi final response type for thread '{thread_id}': {type(final_msg).__name__}")

            if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
                logger.error(f"LLM Engine: Agent ended with unhandled tool calls for thread '{thread_id}': {final_msg.tool_calls}")
                content = "Sorry, I got stuck trying to use a tool."
                return AIMessage(content=content) if return_full else content
            
            return final_msg if return_full else str(final_msg.content)

        except Exception as e:
            logger.error(f"LLM Engine: Error during ask execution for thread '{thread_id}': {e}", exc_info=True)
            error_content = f"Sorry, an error occurred while processing your request with the AI model."
            return AIMessage(content=error_content) if return_full else error_content
# --- END OF FILE engines/llm_logic/langgraph_llm.py (Async _call_model) ---