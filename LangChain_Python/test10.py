import os

from langchain_core.messages import content, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "false"

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",api_key="sk-03e48013fdbf433194b74ff178ea3b92",)

# model.invoke("hello").pretty_print()

messages = [
    HumanMessage(content="你好 我是小明"),
    AIMessage("你好，小明，很高兴认识你"),
    HumanMessage(content="你知道我是谁嘛"),
]

model.invoke(messages).pretty_print()
