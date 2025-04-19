import getpass
import os
from typing import Literal

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_xai import ChatXAI
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.embeddings import init_embeddings

# ─── Setup API Keys ────────────────────────────────────────────────────────────

load_dotenv()

# ─── Define Persistent Memory ─────────────────────────────────────────────────
store = InMemoryStore(index={"dims": 1536, "embed": init_embeddings("google_vertexai:text-multilingual-embedding-002")})
memory = MemorySaver()
manage_mem = create_manage_memory_tool(namespace=("memories",))
search_mem = create_search_memory_tool(namespace=("memories",))

# ─── Define External-World Tools ───────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for fresh data."""
    # TODO: Integrate with a real search API (e.g., Bing, Google)
    return f"[Search results for '{query}']"

@tool
def light_control(action: Literal["on", "off"], location: str) -> str:
    """Control smart lights: 'on' or 'off' in a given location."""
    # TODO: Replace with actual smart-home API calls
    return f"Lights turned {action} in {location}."


# ─── Initialize LLM and Bind Tools ────────────────────────────────────────────
TOOLS = [search_web, light_control, manage_mem, search_mem]
tool_node = ToolNode(TOOLS)

model = ChatXAI(model="grok-2-1212")
bound_model = model.bind_tools(TOOLS)

# ─── Control Flow Helpers ─────────────────────────────────────────────────────
def should_continue(state: MessagesState):
    """Determine whether to invoke a tool next or finish."""
    last_msg = state["messages"][-1]
    # If the AI output included a function call, go to the tool node
    if last_msg.tool_calls:
        return "action"
    # Otherwise, end the graph
    return END


def filter_messages(messages: list):
    """Trim conversation history to fit within context window."""
    # Preserve an initial system prompt if present
    system = messages[0] if isinstance(messages[0], SystemMessage) else None
    # Keep only the last 10 user/assistant turns
    recent = messages[-10:]
    return ([system] + recent) if system else recent


def call_model(state: MessagesState):
    """Invoke the LLM on filtered conversation history."""
    msgs = filter_messages(state["messages"])
    response = bound_model.invoke(msgs)
    return {"messages": response}

# ─── Build the ReAct StateGraph ───────────────────────────────────────────────
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Entry point
workflow.add_edge(START, "agent")
# Loop: agent -> (tool -> agent) or agent -> END
workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["action", END]
)
workflow.add_edge("action", "agent")

# Compile the app with persistent memory
app = workflow.compile(checkpointer=memory, store=store)

# ─── Example Usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Define a persona/system prompt
    system_prompt = SystemMessage(content="You are Lumi, a friendly and knowledgeable AI companion."
                                     "You remember past conversations and can role-play engaging characters.")

    # Start a new conversation session
    messages = [system_prompt, HumanMessage(content="can you turn lights on in kitchen?" )]
    config = {"configurable": {"thread_id": "lumi-session-1"}}

    # Stream the conversation
    for event in app.stream({"messages": messages}, config, stream_mode="values"):
        # Pretty-print each new AI message or tool response
        event["messages"][-1].pretty_print()

    # Later, you can resume by sending more HumanMessage entries to the same thread
