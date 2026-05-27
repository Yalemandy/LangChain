from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# 方法一
class addInput(BaseModel):
    """两数之和"""
    a: int = Field(...,description="第一个参数")
    b: int = Field(...,description="第二个参数")


@tool(args_schema=addInput)
def add(a: int,b: int) -> int:
    return a+b

# 方法二
# @tool
# def add(a: int,b: int) -> int:
#     """两数之和
#
#     Args:
#         a: 第一个整数
#         b: 第二个整数
#     """
#     return a+b

# 方法三
@tool
def add(
        a: Annotated[int,...,"第一个整数"],
        b: Annotated[int,...,"第二个整数"],
) -> int:
    """两数之和

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a+b

print(add.invoke({"a": 2, "b": 3}))