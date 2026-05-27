from typing import List, TypedDict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field

# dp 用不了这个 只有openai才能用
model = ChatOpenAI(model="deepseek-chat",base_url="https://api.deepseek.com",api_key="sk-03e48013fdbf433194b74ff178ea3b92",)

# Pydantic对象 结构化对象
# class Joke(BaseModel):
#     """笑话的结构"""
#     setup: str = Field(description="这是个笑话开头")
#     punchline: str = Field(description="这是个笑话的精彩部分")
#     rating: str = Field(default=None, description="1-10分打分")
#
# class Date(BaseModel):
#     """架构华集合"""
#     jokes: List[Joke]
#

# TypeDict对象
class Jock(TypedDict):
    """笑话的结构"""
    setup: str = Field(description="这是个笑话开头")
    punchline: str = Field(description="这是个笑话的精彩部分")
    rating: str = Field(default=None, description="1-10分打分")



response_model = model.with_structured_output(Jock)
print(response_model.invoke("给我讲个关于唱歌的笑话"))

# print(model.invoke("给我讲个关于唱歌的笑话").content)