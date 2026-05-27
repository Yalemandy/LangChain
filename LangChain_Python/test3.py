import os
from unittest import result

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# gpt_model = init_chat_model("gtp-4o-mini",model_provider="openai")
# deepseek_model = init_chat_model(
#     "deepseek-chat",
#     model_provider="deepseek",
#     api_key = "sk-03e48013fdbf433194b74ff178ea3b92"
# )

message = [
    SystemMessage(content="请补全一段故事，1000字以内"),
    HumanMessage(content="一只狗正在__")
]

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0.7,
    max_tokens=1024,
    configurable_fields=["max_tokens","temperature","model","model_provider"],
    config_prefix="first"
)

result = model.invoke(
    input=message,
    config={
        "configurable":{
            "first_max_tokens": 10,
            "first_model": "deepseek-chat",
            "first_model_provider": "deepseek"
        }
    }
)
print(result.content)

