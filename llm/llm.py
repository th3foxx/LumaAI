from functools import cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.embeddings import init_embeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolNode

from tools.tools import TOOLS
from tools.memory import memory_checkpointer
from settings import settings


llm = ChatOpenAI(
    openai_api_base=settings.ai.openai_api_base,
    openai_api_key=settings.ai.openai_api_key,
    model_name=settings.ai.grok_model,
    temperature=settings.ai.temperature,
)
agent_model = llm.bind_tools(TOOLS)

tool_node = ToolNode(TOOLS)


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


def build_workflow() -> StateGraph:
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["action", END])
    workflow.add_edge("action", "agent")
    return workflow


@cache
def _get_app_and_prompt():
    """
    Открываем PostgresStore один раз за процесс и регистрируем
    корректное закрытие при завершении Python-процесса.
    """
    import atexit                                   #  NEW
    index_cfg = {
        "dims": settings.ai.embedding_dims,
        "embed": init_embeddings(settings.ai.embedding_model),
    }

    # ---------- правильный вход в контекст ----------
    store_cm = PostgresStore.from_conn_string(       # store_cm — контекст-менеджер
        settings.postgres.uri,
        index=index_cfg,
    )
    store = store_cm.__enter__()                     # теперь это настоящий PostgresStore
    atexit.register(store_cm.__exit__, None, None, None)  # красиво закроем на выходе
    # -----------------------------------------------

    store.setup()

    workflow = build_workflow()
    app = workflow.compile(checkpointer=memory_checkpointer, store=store)

    system_prompt = SystemMessage(
        content=(
            "You are Lumi, a friendly and knowledgeable AI companion. "
            "You remember past conversations and can role-play engaging characters. "
            "You have to dialog in Russian."
        )
    )
    return app, system_prompt


def ask_lumi(
    question: str,
    *,
    thread_id: str = "lumi-session-1",
    return_full: bool = False,
) -> str | HumanMessage:

    app, system_prompt = _get_app_and_prompt()

    messages = [system_prompt, HumanMessage(content=question)]
    config = {"configurable": {"thread_id": thread_id}}

    last_event = None
    for event in app.stream(
        {"messages": messages}, config, stream_mode="values"
    ):
        last_event = event

    if last_event is None:
        raise RuntimeError("Lumi did not return any events")

    final_msg = last_event["messages"][-1]
    return final_msg if return_full else final_msg.content