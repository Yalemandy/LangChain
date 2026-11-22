from llama_index.core import SimpleDirectoryReader
import re
import unicodedata
from pathlib import Path
from typing import List, Generator

from llama_index.core import Document
from llama_index.core.llms import ChatMessage,MessageRole
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

import config
import DataClear


# 数据解析函数
def parse_all_formats(input_dir):
    """
    数据解析
    :param input_dir: 文件路径
    :return:  一个document数组
    """

    # 创建加载数据的对象
    reader = SimpleDirectoryReader(
        input_dir= input_dir
    )
    # 加载数据
    documents = reader.load_data()

    return documents

def clean_all_formats(input_dir: Path) -> List[Document]:
    """
    批量清洗
    :param input_dir: 批量处理的路径
    :return: 清洗完成的document列表
    """
    # 将文件解析出来，得到一个Document对象的列表
    dir = parse_all_formats(input_dir)

    # 逐个清洗document对象
    cleaned_docs = [DataClear.clean_doc(doc) for doc in dir]

    # 返回清洗后的对象列表
    return cleaned_docs

def create_qwen_llm(temperature: float=0.5, max_tokens: int=1024) -> DashScope:
    """
    创建通义千问客户端实例
    :return:
    """
    # 获取key
    key = config.dashscope_api_key

    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_PLUS,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# 创建大模型的实例
llm = create_qwen_llm()

def complete(prompt: str) -> str:
    """单次大模型对话"""
    return llm.complete(prompt=prompt).text

def stream_complete(prompt: str) -> Generator[str,None,None]:
    """
    流式打印
    :param prompt:
    :return:
    """
    for chunk in llm.stream_complete(prompt=prompt):
        if chunk.delta:
            yield chunk.delta


def chat(user_prompt: str, sys_prompt: str) -> str:
    """
    多轮对话，非流式
    :param user_prompt:
    :param sys_prompt:
    :return:
    """
    messages = [
        ChatMessage(role=MessageRole.USER, content=user_prompt),
        ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
    ]
    return llm.chat(messages=messages).message.content

def stream_chat(user_prompt: str, sys_prompt: str) -> Generator[ChatMessage,None,None]:
    """
    多轮对话流式输出
    :param user_prompt:
    :param sys_prompt:
    :return:
    """
    messages = [
        ChatMessage(role=MessageRole.USER, content=user_prompt),
        ChatMessage(role=MessageRole.SYSTEM, content=sys_prompt),
    ]

    for chunk in llm.stream_chat(messages=messages):
        if chunk.delta:
            yield chunk.delta


if __name__ == '__main__':

    # print(complete("你是谁"))
    # for text in stream_complete("你是谁"):
    #     print(text,end="",flush=True)
    # print()
    # print(chat("湖北的省会是哪里", "只需要回答城市名即可"))
    for text in stream_chat("你是谁", "简短即可"):
        print(text,end="",flush=True)
    print()






