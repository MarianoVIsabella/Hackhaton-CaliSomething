from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_groq import ChatGroq
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(
    model=os.environ["MODEL"],
    temperature=0,
    disable_streaming=True
)

class State(TypedDict):
    perceptions: str
    reasoning: str
    final_decision: str


def perceive(state: State):
    msg=model.invoke("You are the best news provider about everything that regards markets. Your only job is to retrieve informations " \
    "about news markets to understand what are the prices of every share and the news relating them. Retrieve such news BY ONLY USING THE TOOLS YOU'RE PROVIDED" \
    "If you can't find news or prices by USING")
    return {"perceptions": msg.content}

def reason(state: State):
    msg=model.invoke("")
    return {"reasoning": msg.content}

def act(state: State):
    msg=model.invoke("")
    return {"final_decision": msg.content}

workflow=StateGraph(State)
workflow.add_node("perceive", perceive)
workflow.add_node("reason", reason)
workflow.add_node("act", act)
workflow.add_edge(START,"perceive")
workflow.add_edge("perceive", "reason")
workflow.add_edge("reason", "act")
workflow.add_edge("act", END)

agent= workflow.compile()
agent.invoke()