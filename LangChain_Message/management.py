import os

import tiktoken
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI

api_key = os.environ["DEEPSEEK_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "false"

model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com/v1",api_key=api_key)

# 定义历史消息
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
    HumanMessage(content="What's my name?"),
]

# token_encoder = tiktoken.get_encoding("cl100k_base")

# trim
# 使用 trim_messages 减少发送给模型的消息数量
# trimmer = trim_messages(
#     max_tokens=65,       # 修剪消息的最大令牌数，根据你想要的谈话长度来调整
#     strategy="last",     # 修剪策略：
#                          # “last”（默认）：保留最后的消息。
#                          # “first”：保留最早的消息。
#     token_counter=token_encoder, # 传入一个函数或一个语言模型（因为语言模型有消息令牌计数方法）
#     include_system=True, # 如果想始终保留初始系统消息，可以指定
#     allow_partial=False, # 是否允许拆分消息的内容
#     start_on="human",    # 如果需要确保我们的第一条消息（不包括系统消息）始终是特定类型，可以指定 start_on
# )

trimmer = trim_messages(
    max_tokens=11,       # 修剪消息的最大令牌数，根据你想要的谈话长度来调整
    strategy="last",     # 修剪策略：
                         # “last”（默认）：保留最后的消息。
                         # “first”：保留最早的消息。
    token_counter=len,   # 传入一个函数或一个语言模型（因为语言模型有消息令牌计数方法）
    include_system=True, # 如果想始终保留初始系统消息，可以指定
    allow_partial=False, # 是否允许拆分消息的内容
    start_on="human",    # 如果需要确保我们的第一条消息（不包括系统消息）始终是特定类型，可以指定 start_on
)

print(trimmer.invoke(messages))

# chain = trimmer | model
#
# print(chain.invoke(messages))