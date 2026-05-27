import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# gpt_model = init_chat_model("gtp-4o-mini",model_provider="openai")
# deepseek_model = init_chat_model(
#     "deepseek-chat",
#     model_provider="deepseek",
#     api_key = "sk-03e48013fdbf433194b74ff178ea3b92"
# )
#
# print(deepseek_model.invoke("你是谁").content)

message = [
    SystemMessage(content="帮我翻译，将中文翻译成英文"),
    HumanMessage(content="你好")
]
#
#
# 模型模拟器
config_model = init_chat_model(temperature=0.4)

print(config_model.invoke(input=message, config={"configurable": {"model": "deepseek-chat"}}).content)


