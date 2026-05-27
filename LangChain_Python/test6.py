from typing import Tuple, List

from langchain_core.tools import StructuredTool

# 方法一
# def add(a: int, b: int) -> int:
#     """两数之和"""
#     return a + b
#
# add_tool = StructuredTool.from_function(func=add)

# 方法二
# def add(a: int, b: int) -> int:
#     return a + b
#
# add_tool = StructuredTool.from_function(
#     func=add,
#     name="ADD",                # 工具名
#     description="两数之和",      # 工具描述
# )

# 方法三
def add(a: int, b: int) -> Tuple[str,List[int]]:
    nums = [a, b]
    content = f"{nums}两数之和的结果是{a + b}"
    return content, nums

add_tool = StructuredTool.from_function(
    func=add,
    name="ADD",                # 工具名
    description="两数之和",      # 工具描述
    response_format="content_and_artifact"
)

# 模拟大模型调用姿势
print(add_tool.invoke(
    {
        "name": "ADD",
        "args": {"a": 1, "b": 2},
        "type": "tool_call",  # 必填
        "id": "1",            # 必填  将工具调用请求和调用结果关联起来
    }
))

# print(add_tool.invoke({"a": 2, "b": 3}))
# print(add_tool.name)- 78
# print(add_tool.description)