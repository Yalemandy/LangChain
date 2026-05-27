from itertools import chain
from unittest import result

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-flash",
    # 采样温度 温度越高 回复越天马行空 反之 回复越保守  范围: 0-1/2 open支持0-2
    # temperature=0,
    # 最大token数
    # max_tokens=None,
    # 超时时间
    # timeout=None,
    # 最大重新测试次数
    # max_retries=2,
    # 大模型的key密钥
    api_key="sk-03e48013fdbf433194b74ff178ea3b92",
    base_url="https://api.deepseek.com",
    # organization="...",
    # other params...
)


# SystemMessage 系统消息
# HumanMessage 用户消息
message = [
    SystemMessage(content="帮我翻译，将中文翻译成英文"),
    HumanMessage(content="你好")
]

# 调用大模型LLM
result = model.invoke(message)
print(result)

# 定义输出解析器
parser = StrOutputParser()
# print(parser.invoke(result))

# 定义链 | 管道
chain = model | parser

# 执行链
print(chain.invoke(message))