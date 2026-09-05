from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 1. 定义输⼊模式 - 只包含⽤⼾问题
class InputState(TypedDict):
    question: str

# 2. 定义输出模式 - 只包含AI答案
class OutputState(TypedDict):
    answer: str

# 3. 定义完整状态模式（内部使⽤）
class OverallState(InputState, OutputState):
    pass

def answer_node(state: InputState):
    """处理输⼊并⽣成答案"""
    # 这⾥可以访问 question，⽣成 answer
    return {
        "answer": f"Answer to: {state['question']}",
        "question": state["question"]
    }

# 构建图时指定输⼊输出模式
builder = StateGraph(
    OverallState,
    input_schema=InputState,
    # 输⼊验证
    output_schema=OutputState
    # 输出过滤
)
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()
# 测试
result = graph.invoke({"question": "What is LangGraph?"})
print(result)

