import operator
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Overwrite


class Message(TypedDict):
    messages: Annotated[list[str], operator.add]

def node1(state: Message):
    return {
        "messages": ["1111"]
    }

def node2(state: Message):
    return {
        "messages": Overwrite(["2222"])
    }

graph = StateGraph(Message)

graph.add_node("node1", node1)
graph.add_node(node2)

graph.add_edge(START, "node1")
graph.add_edge("node1","node2")
graph.add_edge("node2",END)

result = graph.compile()

print(result.invoke({"messages": ["3333"]})["messages"])
