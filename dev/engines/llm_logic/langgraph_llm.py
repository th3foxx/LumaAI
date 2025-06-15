# langgraph_llm.py

import os
import logging
import asyncio
from typing import Annotated, Union, Dict, Any, List, Optional, Callable, Coroutine

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
)
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict

from settings import AISettings
from tools import TOOLS as project_tools
from engines.llm_logic.base import LLMLogicEngineBase

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONVERSATION_MESSAGES = 20
MIN_CONVERSATION_MESSAGES = 8

class LangGraphLLMEngine(LLMLogicEngineBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ai_settings: AISettings = config["ai_settings"]
        
        # --- ИСПРАВЛЕНИЕ: Удалены устаревшие проверки LTM-атрибутов ---
        # Эти проверки больше не нужны, так как движок не отвечает за LTM.

        self.graph = None
        self.memory = None
        self.llm_with_tools = None
        self.max_conversation_messages: int = DEFAULT_MAX_CONVERSATION_MESSAGES
        
        self.system_prompt_content: str = self.ai_settings.system_prompt 
        logger.info(f"Using system prompt: '{self.system_prompt_content[:100]}...'")

        if not self.ai_settings.openai_api_base:
            logger.error("OPENAI_API_BASE is not set. LangGraphLLMEngine requires it for the LLM.")
            raise ValueError("OPENAI_API_BASE is required for LangGraphLLMEngine.")

        api_key_to_use = self.ai_settings.openai_api_key
        if not api_key_to_use:
            logger.warning(
                f"OPENAI_API_KEY is not set. This may cause issues if the endpoint "
                f"'{self.ai_settings.openai_api_base}' requires authentication."
            )

        configured_max_messages = getattr(self.ai_settings, 'max_conversation_messages', DEFAULT_MAX_CONVERSATION_MESSAGES)
        if configured_max_messages < MIN_CONVERSATION_MESSAGES:
            logger.warning(
                f"Configured 'max_conversation_messages' ({configured_max_messages}) is less than "
                f"the minimum allowed ({MIN_CONVERSATION_MESSAGES}). "
                f"Using minimum value: {MIN_CONVERSATION_MESSAGES}."
            )
            self.max_conversation_messages = MIN_CONVERSATION_MESSAGES
        else:
            self.max_conversation_messages = configured_max_messages
        
        logger.info(f"Conversation history will be pruned to a maximum of {self.max_conversation_messages} messages. LLM will see up to {self.max_conversation_messages -1} messages (plus system prompt).")

        logger.info(
            f"Initializing ChatOpenAI for LangGraphLLMEngine with: "
            f"Model='{self.ai_settings.grok_model}', "
            f"Base URL='{self.ai_settings.openai_api_base}', "
            f"Temperature='{self.ai_settings.temperature}'"
        )

        try:
            llm = ChatOpenAI(
                model=self.ai_settings.grok_model,
                api_key=api_key_to_use,
                base_url=self.ai_settings.openai_api_base,
                temperature=self.ai_settings.temperature,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
            raise ValueError(f"Could not initialize LLM for LangGraph: {e}") from e

        self.llm_with_tools = llm.bind_tools(project_tools if project_tools else [])
        if project_tools:
            logger.info(f"LLM bound with {len(project_tools)} tools.")
        else:
            logger.info("LLM initialized without any tools.")

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chatbot_node_func)
        
        tool_node = ToolNode(project_tools if project_tools else [])
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")

        self.memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=self.memory)
        logger.info("LangGraphLLMEngine initialized successfully with tools, memory-based checkpointer, message pruning, and system prompt.")


    def _chatbot_node_func(self, state: State) -> Dict[str, List[BaseMessage]]:
        current_messages_from_state: List[BaseMessage] = state["messages"]
        logger.debug(f"Chatbot node: {len(current_messages_from_state)} messages in state. Max set to {self.max_conversation_messages}.")

        messages_to_update_state: List[BaseMessage] = []
        
        num_history_messages_for_llm = self.max_conversation_messages - 1
        if num_history_messages_for_llm < 0:
            num_history_messages_for_llm = 0

        actual_history_llm_input_size = min(len(current_messages_from_state), num_history_messages_for_llm)
        
        history_for_llm_input: List[BaseMessage] = []
        if actual_history_llm_input_size > 0:
            history_for_llm_input = current_messages_from_state[-actual_history_llm_input_size:]
            
            if history_for_llm_input and isinstance(history_for_llm_input[0], ToolMessage):
                orphaned_tool_msg = history_for_llm_input[0]
                logger.warning(
                    f"Orphaned ToolMessage (tool_call_id: {orphaned_tool_msg.tool_call_id}, "
                    f"name: {getattr(orphaned_tool_msg, 'name', 'N/A')}) "
                    f"would be the first history message in LLM input. Removing it."
                )
                history_for_llm_input = history_for_llm_input[1:]
                        
                if not history_for_llm_input and len(current_messages_from_state) > 1:
                    logger.warning("history_for_llm_input became empty after removing orphaned ToolMessage. Taking last message from history as fallback.")
                    history_for_llm_input = [current_messages_from_state[-1]]
        
        elif len(current_messages_from_state) > 0 and num_history_messages_for_llm == 0 :
            logger.warning(
                f"Calculated history LLM input size is 0 (max_messages={self.max_conversation_messages}). "
                f"Sending the last message from state if available."
            )
            history_for_llm_input = [current_messages_from_state[-1]]

        messages_for_llm_input = [SystemMessage(content=self.system_prompt_content)] + history_for_llm_input
        
        if (len(current_messages_from_state) + 1) > self.max_conversation_messages:
            num_to_remove_from_state = (len(current_messages_from_state) + 1) - self.max_conversation_messages
            if num_to_remove_from_state > 0:
                messages_to_be_deleted = current_messages_from_state[:num_to_remove_from_state]
                remove_message_ops = [
                    RemoveMessage(id=m.id) for m in messages_to_be_deleted if hasattr(m, 'id') and m.id is not None
                ]
                messages_to_update_state.extend(remove_message_ops)
                logger.debug(f"History Pruning: {len(remove_message_ops)} oldest messages marked for removal from state.")
        
        logger.debug(f"Invoking LLM with {len(messages_for_llm_input)} messages (incl. system prompt). History snippet: {[str(type(m)) + ': ' + str(m.content)[:70] + (' (tool_calls)' if isinstance(m, AIMessage) and m.tool_calls else '') + (f'(tool_id:{m.tool_call_id})' if isinstance(m, ToolMessage) else '') for m in history_for_llm_input]}")

        if not history_for_llm_input and len(current_messages_from_state) > 0:
             logger.error("Critical: history_for_llm_input is empty, but current_messages_from_state was not. This indicates a logic flaw in history preparation.")
             messages_for_llm_input = [SystemMessage(content=self.system_prompt_content), current_messages_from_state[-1]]
        elif not history_for_llm_input and len(current_messages_from_state) == 0:
            logger.warning("Chatbot node: history_for_llm_input is empty because current_messages_from_state is empty. Sending only system prompt.")

        response_message = self.llm_with_tools.invoke(messages_for_llm_input)
        messages_to_update_state.append(response_message)
        
        return {"messages": messages_to_update_state}

    async def ask(self, question: str, thread_id: str, return_full: bool = False) -> Union[str, BaseMessage, Dict[str, Any]]:
        if not self.graph:
            logger.error("LangGraph graph is not compiled. Cannot process 'ask' request.")
            return {"response": "Error: LLM Engine is not properly initialized.", "ltm_messages": []}

        graph_config = {"configurable": {"thread_id": thread_id}}
        
        input_message = HumanMessage(content=question)
        input_data = {"messages": [input_message]}

        logger.debug(f"Invoking LangGraph for thread_id='{thread_id}' with new question.")
        
        try:
            # ainvoke выполняет граф и возвращает конечное состояние
            final_state_dict = await self.graph.ainvoke(input_data, graph_config)
        except Exception as e:
            logger.error(f"Error during LangGraph ainvoke for thread_id='{thread_id}': {e}", exc_info=True)
            return {"response": f"Sorry, an error occurred while I was thinking: {str(e)}", "ltm_messages": []}

        # Получаем ПОЛНЫЙ обновленный список сообщений из состояния.
        # В нем содержится вся история, включая только что добавленные сообщения этого хода.
        all_messages: List[BaseMessage] = final_state_dict.get("messages", [])
        
        # Находим последнее сообщение от AI для ответа пользователю.
        last_ai_message: Optional[AIMessage] = None
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content: # Ищем сообщение с текстовым контентом
                last_ai_message = msg
                break
        
        messages_for_ltm = all_messages
        logger.debug(f"Returning full state with {len(messages_for_ltm)} messages for potential LTM processing.")

        result = {
            "response": "I'm sorry, I couldn't formulate a response.",
            "ltm_messages": messages_for_ltm
        }

        if last_ai_message:
            logger.info(f"LLM response for thread_id='{thread_id}' successfully retrieved. Total messages in state: {len(all_messages)}.")
            result["response"] = last_ai_message if return_full else last_ai_message.content
        else:
            logger.warning(f"No AIMessage with content found in the final state for thread_id='{thread_id}'.")

        return result

    async def startup(self):
        logger.info("LangGraphLLMEngine startup: No specific actions needed for MemorySaver.")
        pass

    async def shutdown(self):
        logger.info("LangGraphLLMEngine shutdown: No specific actions needed for MemorySaver.")
        pass