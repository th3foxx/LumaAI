import atexit
import logging
from functools import cache
from typing import List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.embeddings import init_embeddings # Assuming this exists and works
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore # Use BaseStore for type hint
from langgraph.store.postgres import PostgresStore # Keep specific import for instantiation
from langgraph.prebuilt import ToolNode

from tools.tools import TOOLS
from tools.memory import memory_checkpointer # Assuming this is correct
from settings import settings

logger = logging.getLogger(__name__)

# Global store variable to manage its lifecycle
_store_cm = None
_store: Optional[BaseStore] = None

def get_store() -> BaseStore:
    """Gets the initialized PostgresStore instance."""
    global _store, _store_cm
    if _store is None:
        logger.info(f"Initializing PostgresStore with URI: {settings.postgres.uri.split('@')[-1]}") # Log URI without credentials
        index_cfg = {
            "dims": settings.ai.embedding_dims,
            "embed": init_embeddings(settings.ai.embedding_model), # Ensure this handles potential errors
        }
        try:
            # Use context manager for setup/teardown
            _store_cm = PostgresStore.from_conn_string(
                settings.postgres.uri,
                index=index_cfg,
            )
            _store = _store_cm.__enter__() # Enter context to get the store instance
            _store.setup() # Ensure tables/indexes are ready
            logger.info("PostgresStore initialized and setup complete.")
            # Register the exit handler *after* successful initialization
            atexit.register(close_llm_resources)
        except Exception as e:
            logger.error(f"Failed to initialize or setup PostgresStore: {e}", exc_info=True)
            # Prevent app from starting if DB connection fails? Or run without memory?
            raise RuntimeError("Could not connect to or setup the LangGraph PostgresStore.") from e
    return _store

def close_llm_resources():
    """Closes the PostgresStore connection via its context manager."""
    global _store_cm, _store
    if _store_cm:
        logger.info("Closing PostgresStore connection...")
        try:
            _store_cm.__exit__(None, None, None) # Exit the context manager
            logger.info("PostgresStore connection closed.")
        except Exception as e:
            logger.error(f"Error closing PostgresStore: {e}", exc_info=True)
        _store = None
        _store_cm = None

# Initialize LLM and bind tools
# Consider adding error handling for ChatOpenAI initialization
try:
    llm = ChatOpenAI(
        openai_api_base=settings.ai.openai_api_base,
        openai_api_key=settings.ai.openai_api_key,
        model_name=settings.ai.grok_model,
        temperature=settings.ai.temperature,
        # Add timeouts? e.g., request_timeout=30
    )
    agent_model = llm.bind_tools(TOOLS)
    logger.info(f"ChatOpenAI initialized with model: {settings.ai.grok_model}")
except Exception as e:
    logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
    # Decide if the application can run without LLM
    raise RuntimeError("Could not initialize the LLM.") from e


tool_node = ToolNode(TOOLS)

def should_continue(state: MessagesState) -> str:
    """Determines whether to continue with tools or end."""
    # Check for presence of tool calls in the last message
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and last_msg.tool_calls:
         # Check if tool calls are empty (sometimes happens with certain models/prompts)
         if not any(tc.get("name") for tc in last_msg.tool_calls):
              logger.warning("Agent returned tool_calls, but they are empty. Ending turn.")
              return END
         logger.info(f"Agent requested tool(s): {[tc.get('name') for tc in last_msg.tool_calls]}")
         return "action"
    else:
         logger.info("Agent did not request tools. Ending turn.")
         return END


def filter_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Filters messages to keep system prompt and recent history."""
    if not messages:
        return []
    # Ensure the first message is SystemMessage if it exists
    has_system = isinstance(messages[0], SystemMessage)
    system = messages[0] if has_system else None
    # Get the last N messages (excluding system if present)
    history_start_index = 1 if has_system else 0
    recent = messages[max(history_start_index, len(messages) - settings.ai.history_length):] # Use setting for history length

    filtered = []
    if system:
        filtered.append(system)
    filtered.extend(recent)
    # logger.debug(f"Filtered messages count: {len(filtered)}")
    return filtered


def call_model(state: MessagesState) -> MessagesState:
    """Invokes the agent model with filtered messages."""
    # Filter messages before sending to the model
    filtered_msgs = filter_messages(state["messages"])
    if not filtered_msgs:
         logger.warning("call_model received empty state messages after filtering.")
         # Return empty response or raise error? For now, return state as is.
         return state # Or {"messages": []} ? Check LangGraph behavior

    logger.debug(f"Calling agent model with {len(filtered_msgs)} messages.")
    # Add error handling around invoke
    try:
        response = agent_model.invoke(filtered_msgs)
        # logger.debug(f"Agent model response: {response}")
        # Ensure response is appended correctly
        # LangGraph expects a dictionary mapping to state keys
        return {"messages": [response]} # Return only the new message to be appended
    except Exception as e:
         logger.error(f"Error invoking agent model: {e}", exc_info=True)
         # How to handle LLM errors? Return an error message? Retry?
         # For now, re-raise or return an error message in the state
         # Returning an error message:
         error_msg = HumanMessage(content=f"Sorry, I encountered an error: {e}")
         return {"messages": [error_msg]}
         # Or re-raise: raise e


def build_workflow() -> StateGraph:
    """Builds the LangGraph workflow."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", END: END} # Map return values to node names
    )
    workflow.add_edge("action", "agent") # Loop back to agent after action
    logger.info("LangGraph workflow built.")
    return workflow


# Use @cache to ensure the app and prompt are created only once
@cache
def _get_app_and_prompt() -> tuple[any, SystemMessage]:
    """Initializes and returns the compiled LangGraph app and system prompt."""
    logger.info("Initializing LangGraph app and system prompt...")
    # Get the store (which initializes on first call)
    store = get_store()

    # Get the checkpointer (assuming memory_checkpointer uses the store)
    checkpointer = memory_checkpointer # Pass the store if needed: memory_checkpointer(store)

    workflow = build_workflow()
    # Compile the graph with checkpointer and store
    app = workflow.compile(checkpointer=checkpointer, store=store) # Pass store here if checkpointer doesn't handle it
    logger.info("LangGraph app compiled.")

    system_prompt = SystemMessage(
        content=(
            "You are Lumi, an efficient AI voice assistant. Your primary goal is to fulfill the user's request quickly and accurately. "
            "You remember the last few turns of this conversation."
            "Always respond in Russian unless asked otherwise."

            "First, analyze the user's request. "
            "DECIDE QUICKLY: Can you answer directly based on the conversation history or your general knowledge? "
            "If YES, provide a concise answer immediately without using tools. "
            "If NO, and the request requires external action (like controlling a device, searching memory, etc.) or information you don't have, IDENTIFY the necessary tool and its arguments. "

            "When using a tool: Briefly state *what* you need the tool for (e.g., 'Checking device status...', 'Looking up notes...'), then call the tool immediately. "
            "After the tool runs, use its output to formulate your final, concise response to the user."

            "Keep your spoken responses natural and brief."
        )
    )
    logger.info("System prompt created.")
    return app, system_prompt


async def ask_lumi(
    question: str,
    *,
    thread_id: str = "lumi-default-thread", # Default thread ID
    return_full: bool = False,
) -> Union[str, BaseMessage]:
    """
    Asks the Lumi agent a question using the LangGraph app.

    Args:
        question: The user's question or statement.
        thread_id: The conversation thread ID.
        return_full: Whether to return the full message object or just content.

    Returns:
        The agent's response content (str) or the full message object.

    Raises:
        RuntimeError: If the agent fails to produce a response.
    """
    logger.info(f"ask_lumi called for thread '{thread_id}'. Question: '{question[:50]}...'")
    try:
        app, system_prompt = _get_app_and_prompt() # Gets cached app/prompt

        # Prepare input messages
        messages = [system_prompt, HumanMessage(content=question)]
        config = {"configurable": {"thread_id": thread_id}}

        # Use ainvoke for async execution, stream values for intermediate results
        # final_state = await app.ainvoke({"messages": messages}, config=config)

        # Streaming is better for seeing intermediate steps if needed, but invoke is simpler for just the final result
        # Let's stick to stream as it was originally used, assuming potential value in observing steps
        final_event = None
        async for event in app.astream(
            {"messages": messages}, config=config, stream_mode="values"
        ):
            # logger.debug(f"LangGraph stream event for thread '{thread_id}': {event}")
            final_event = event # Keep track of the last event which holds the final state

        if final_event is None or not final_event.get("messages"):
            logger.error(f"Lumi agent did not return any events or messages for thread '{thread_id}'.")
            raise RuntimeError("Lumi did not return a valid response.")

        # Get the last message from the final state
        final_msg = final_event["messages"][-1]
        logger.info(f"Lumi final response type for thread '{thread_id}': {type(final_msg).__name__}")

        # Ensure the final message is not a tool call request that wasn't handled (shouldn't happen with correct graph)
        if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
             logger.error(f"Agent ended with unhandled tool calls for thread '{thread_id}': {final_msg.tool_calls}")
             # Return an error message or the raw tool call?
             return "Sorry, I got stuck trying to use a tool." if not return_full else final_msg


        return final_msg if return_full else str(final_msg.content)

    except Exception as e:
        logger.error(f"Error during ask_lumi execution for thread '{thread_id}': {e}", exc_info=True)
        # Re-raise or return a user-friendly error message?
        # Re-raising allows the caller (background task) to handle it.
        raise # Re-raise the exception