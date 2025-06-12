from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from davia import Davia
import sqlite3
import datetime
import uuid

# Load environment variables
load_dotenv(dotenv_path='../.env', override=True)

# Initialize Davia
app = Davia(title="LangGraph Chatbot with Memory & Tools")

# Persistent memory
sqlite_connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_connection)

# Define tools
search = GoogleSerperAPIWrapper()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time in the given format."""
    return datetime.datetime.now().strftime(format)

tools = [
    get_system_time,
    Tool(
        name="google_search",
        func=search.run,
        description="Search the web using Google Serper API."
    )
]

# Model with tools
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
llm_with_tools = llm.bind_tools(tools=tools)

# LangGraph State
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

# LangGraph nodes
def chatbot(state: BasicChatState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: BasicChatState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

# Tool node and graph setup
tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

# Compile the graph with memory
langgraph_app = graph.compile(checkpointer=memory)

# LangGraph UI with proper configuration
@app.graph
def run_chatbot_graph():
    """LangGraph chatbot with memory and tools."""
    # Return the compiled graph with default configuration
    return langgraph_app

# Alternative: If you need to specify configuration
@app.task
def chat_with_memory(user_input: str, thread_id: str = None) -> str:
    """Chat with memory and tools using a specific thread ID."""
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    try:
        result = langgraph_app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# Simple task UI (fallback for clickable form)
@app.task
def prompt_chat(user_input: str) -> str:
    """Chat with memory and tools using a single input."""
    try:
        result = langgraph_app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": "default_session"}}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# Streaming version (if Davia supports it)
@app.task
def stream_chat(user_input: str) -> str:
    """Stream chat responses."""
    try:
        responses = []
        for chunk in langgraph_app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": "stream_session"}}
        ):
            if "messages" in chunk:
                responses.append(chunk["messages"][-1].content)
        
        return responses[-1] if responses else "No response generated"
    except Exception as e:
        return f"Error: {str(e)}"

# Health check function
@app.task
def health_check() -> str:
    """Check if the system is working."""
    try:
        test_result = langgraph_app.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            config={"configurable": {"thread_id": "health_check"}}
        )
        return "System is healthy!"
    except Exception as e:
        return f"System error: {str(e)}"

# Note: When using Davia CLI, remove the if __name__ == "__main__" block
# The app instance should be available at module level for the CLI to find it

# If you want to run directly with Python (not recommended), keep this:
# if __name__ == "__main__":
#     app.run()