from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode



load_dotenv()

class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def sum(a: int, b: int) -> int:
    """THis Add two numbers"""

    return a+ b

@tool
def subtract(a: int, b: int) -> int:
    """THis Subtract two numbers"""

    return a - b


@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [sum, subtract, multiply]

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6
)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )

    response = model.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]

    if not messages:
        return "continue"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    return "end"



graph = StateGraph(AgentState)

graph.add_node("our_Agent", model_call)

tool_node = ToolNode(tools=tools)

graph.add_node(tool_node)

graph.set_entry_point("our_Agent")

graph.add_conditional_edges(
    "our_Agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_Agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
