import os
from xml.etree.ElementInclude import include

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, filter_messages, merge_message_runs
from langchain_openai import ChatOpenAI

api_key = os.environ["DEEPSEEK_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "false"

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com/v1",api_key=api_key)

# 历史消息记录
messages = [
    SystemMessage("你是一个聊天助手。"),
    SystemMessage("你总是以笑话回应。"),
    HumanMessage("为什么要使用 LangChain?"),
    HumanMessage("为什么要使用 LangGraph?"),
    AIMessage("因为当你试图让你的代码更有条理时，LangGraph 会让你感到“节点”是个好主意！"),
    AIMessage("不过别担心，它不会“分散”你的注意力！"),
    HumanMessage("选择LangChain还是LangGraph?"),
]

merge = merge_message_runs()
print(merge)

chain = merge | model
chain.invoke(messages).pretty_print()
