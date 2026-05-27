from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated


@tool
def add(
        a: Annotated[int, "第一个整数"],
        b: Annotated[int, "第二个整数"],
) -> int:
    """两数之和"""
    return a + b

@tool
def multiply(
        a: Annotated[int, "第一个整数"],
        b: Annotated[int, "第二个整数"],
) -> int:
    """两数相乘"""
    return a * b

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",api_key="sk-03e48013fdbf433194b74ff178ea3b92",)

# 绑定工具
tools = [add, multiply]
# model_with_tools = model.bind_tools(tools=tools)
# 强制选择工具
model_with_tools = model.bind_tools(tools=tools, tool_choice="any")

# 定义消息列表，添加要传递给聊天模型的信息  问题
message = [
    HumanMessage("8乘9等于几？,3加9等于多少？")
]

# 选择AImessage 选择工具以及工具的调用姿势
ai_message = model_with_tools.invoke(message)

# 添加到这个message里面
message.append(ai_message)

# ai理解并调用工具 然后再将结果也放到message里面
for tool_call in ai_message.tool_calls:
    seleated_tool = {"add":add, "multiply":multiply}[tool_call["name"].lower()]
    tool_message = seleated_tool.invoke(tool_call)
    message.append(tool_message)

# 整合发给ai聊天模型
print(message)
print(model.invoke(message).content)
# print("完整响应对象：", response)
# print("是否有 tool_calls：", ai_message.tool_calls)
# print("tool_calls 内容：", response.tool_calls if hasattr(response, 'tool_calls') else "不存在")