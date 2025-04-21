from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, List
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.embeddings import init_embeddings
from langchain_xai import ChatXAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.postgres import PostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

__all__ = ["main"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from .env into the process environment


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    postgres_user: str = os.getenv("POSTGRES_USER", "lumi")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")
    postgres_db: str = os.getenv("POSTGRES_DB", "lumi")
    embedding_model: str = os.getenv(
        "EMBED_MODEL", "google_vertexai:text-multilingual-embedding-002"
    )
    embedding_dims: int = int(os.getenv("EMBED_DIMS", 768))
    grok_model: str = os.getenv("GROK_MODEL", "grok-2-1212")

    @property
    def postgres_uri(self) -> str:
        """Return a fully quoted PostgreSQL URI with safe password handling."""

        pwd = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{pwd}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()

# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

memory_checkpointer = MemorySaver()

manage_mem = create_manage_memory_tool(namespace=("memories",), instructions="""
Proactively call this tool when you:

1. Learn or confirm personal details about the user:
   - Name or nickname, gender, age.
   - Profession, occupation, location (city, country).
   - Special circumstances (e.g. “I’m traveling,” “I’ve just moved,” “I’m studying for exams”).

2. Detect stable preferences, habits, or styles:
   - Favorite topics, genres, products, tastes.
   - Communication style: formal vs. casual, use of emojis, humor.
   - Productivity patterns: “I’m most alert in the morning,” “Evenings are for brainstorming.”

3. Receive an explicit request to remember something:
   - “Please remember …,” “Don’t forget …,” “Remind me later about ….”
   - Reminders for meetings, events, tasks or future conversational prompts.

4. Encounter a long‑term project or ongoing context:
   - Trip planning, multi‑step assignments, home renovations.
   - Follow‑up on previously discussed items when the user returns to the conversation.

5. Uncover new, important contextual details mid‑dialogue:
   - Additional constraints, criteria, or project requirements.
   - Status updates (e.g. “My draft is now complete,” “The budget changed”).

6. Identify that an existing memory is incorrect or outdated:
   - The user corrects previously stored facts.
   - You notice contradictions between stored memory and new information.

7. Need to leverage past interactions to improve response quality:
   - Address the user by name, reference earlier jokes or examples.
   - When a key fact from prior chats adds coherence or personalization.

—
Do not call this tool for:
- Ephemeral or one‑off details unlikely to matter later (e.g. “I had an apple today”).
- Non‑personal trivia that won’t influence future behavior.
""")

search_mem = create_search_memory_tool(
    namespace=("memories",),
    instructions="""
Proactively call this tool when you need to retrieve stored user memory because it’s relevant to the current response or decision‑making, especially if the information may have fallen out of your short‑term context (which spans only the last 10 turns). For example:

1. Addressing the user by name, nickname, or title.
2. Applying known preferences (topics, styles, favorite products, habits).
3. Referring back to ongoing projects or long‑term plans (trip itinerary, coursework, renovation).
4. Reusing previous constraints or criteria (budget limits, deadlines, personal requirements).
5. Ensuring consistency when the user revisits a topic after a gap longer than 10 turns.
6. Checking for updates or corrections before acting on a stored fact.
7. Personalizing your tone or suggestions based on earlier details (time zone, communication style).
8. Verifying whether a memory exists before prompting the user again (“As I recall, you prefer X—does that still hold?”).

—  
**Do not call** this tool if:
- The needed details are still present in your immediate 10‑turn context.
- The information is ephemeral or truly one‑off and not stored.
"""
)

# ---------------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------------


@tool
def search_web(query: str) -> str:
    """Search the web for fresh data (placeholder)."""
    return f"[Search results for '{query}']"


@tool
def light_control(action: Literal["on", "off"], location: str) -> str:
    """Control smart lights: 'on' or 'off' in a given location (placeholder)."""
    return f"Lights turned {action} in {location}."


TOOLS: List = [search_web, light_control, manage_mem, search_mem]

tool_node = ToolNode(TOOLS)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

llm = ChatXAI(model=settings.grok_model)
agent_model = llm.bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    return "action" if last_msg.tool_calls else END


def filter_messages(messages: list) -> list:
    system = messages[0] if isinstance(messages[0], SystemMessage) else None
    recent = messages[-10:]
    return ([system] + recent) if system else recent


def call_model(state: MessagesState):
    msgs = filter_messages(state["messages"])
    response = agent_model.invoke(msgs)
    return {"messages": response}

# ---------------------------------------------------------------------------
# Graph assembly (uncompiled)
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["action", END])
    workflow.add_edge("action", "agent")
    return workflow

# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    index_cfg = {
        "dims": settings.embedding_dims,
        "embed": init_embeddings(settings.embedding_model),
    }

    with PostgresStore.from_conn_string(settings.postgres_uri, index=index_cfg) as store:
        store.setup()

        workflow = build_workflow()
        app = workflow.compile(checkpointer=memory_checkpointer, store=store)

        system_prompt = SystemMessage(
            content=(
                "You are Lumi, a friendly and knowledgeable AI companion. "
                "You remember past conversations and can role‑play engaging characters."
                "You have to dialog in Russian."
            )
        )

        messages = [system_prompt, HumanMessage(content="сколько мне лет?")]
        config = {"configurable": {"thread_id": "lumi-session-1"}}

        for event in app.stream({"messages": messages}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Goodbye!")
