import os

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

api_key = os.environ["DEEPSEEK_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "false"

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com/v1",api_key=api_key)

# model.invoke("我是小闵").pretty_print()
# model.invoke("你知道我是谁嘛").pretty_print()

# message = [
#     SystemMessage(content="你是小余，你喜欢小闵"),
#     HumanMessage(content="我是小闵你好"),
#     AIMessage(content="你好啊"),
#     HumanMessage(content="你知道我是谁嘛,你是谁啊，你有喜欢的人嘛")
# ]

# model.invoke(message).pretty_print()


store = {}
# 根据会话id 来查询会话中的消息列表
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        # InMemoryChatMessageHistory 帮我们将AiMessage、HumanMessage。。。被他自动注入进去了
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 包装了model 让model拥有了存储历史消息的能力
with_history_message_model = RunnableWithMessageHistory(model,get_session_history)

# model: Runnable实例
# invoke: config配置runnable实例
config={"configurable":{"session_id":"1"}}
with_history_message_model.invoke(
    [HumanMessage(content="我是小明")],
    config=config,
).pretty_print()

with_history_message_model.invoke(
    [HumanMessage(content="你知道我是谁嘛")],
    config=config,
).pretty_print()


















