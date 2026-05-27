import os
from xml.etree.ElementInclude import include

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, filter_messages
from langchain_openai import ChatOpenAI

api_key = os.environ["DEEPSEEK_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "false"

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com/v1",api_key=api_key)

# 历史消息
messages = [
    SystemMessage("你是一个聊天助手", id="1"),
    HumanMessage("示例输入", id="2"),
    AIMessage("示例输出", id="3"),
    HumanMessage("真实输入", id="4"),
    AIMessage("真实输出", id="5"),
]

# 按照类型筛选
# print(filter_messages(include_types="human").invoke(messages))

# print(filter_messages(messages, include_types="human"))

# 通过id筛选
# print(filter_messages(messages, exclude_ids=["3"]))

# 按照类型+id筛选
print(filter_messages(messages, exclude_ids=["3"],include_types=[HumanMessage]))

