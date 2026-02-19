from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):

    message: List[HumanMessage]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6
)

def process(state: AgentState) -> AgentState:

    response = llm.invoke(state['message'])
    print(f"\n AI : {response.content}")
    return state

graph = StateGraph(AgentState)


graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge( "process", END)
agent = graph.compile()

user_input = input("Enter your message....")
while user_input.lower() != "exit":
    agent.invoke({"message": [HumanMessage(content= user_input)]})
    user_input = input("Enter your message....")
