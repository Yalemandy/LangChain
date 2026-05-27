from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import tavily_search, TavilySearch

from test1 import message

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",api_key="sk-03e48013fdbf433194b74ff178ea3b92",)

# 定义工具
tool = TavilySearch(max_results=4)

# 绑定工具
model_with_tools = model.bind_tools([tool])

# 定义消息列表
message = [
    HumanMessage("丽水天气如何")
]

ai_message = model_with_tools.invoke(message)

message.append(ai_message)

for tool_call in ai_message.tool_calls:
    tool_message = tool.invoke(tool_call)
    message.append(tool_message)

print(model.invoke(message).content)