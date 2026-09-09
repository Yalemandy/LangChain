# 提示链模式
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

model = init_chat_model("DeepSeek-V4-Flash", model_provider="deepseek")


# 1. 定义输⼊模式 - 只包含⽤⼾输⼊
# ⽤⼾输⼊的主题
class InputState(TypedDict):
    topic: str

# 2. 定义输出模式 - 只包含最终结果
# 最终的内容
class OutputState(TypedDict):
    final_content: str

# 3. 定义完整状态模式（内部使⽤）
class OverallState(InputState, OutputState):
    # 第⼀步：⽣成的⼤纲
    outline: str
    # 第⼆步：⽣成的初稿
    draft: str
    # 第三步：润⾊后的稿件
    polished_draft: str

# 节点
PROMPT_1 = (
    "根据主题⽣成⽂章⼤纲。\n"
    "主题：{topic}\n"
    "要求："
    "1.只需两个最核⼼标题"
    "2.不⽤进⾏说明，只返回最终⼤纲"
)
def generate_outline(state: InputState):
    """生成大纲"""
    prompt = PROMPT_1.format(topic=state["topic"])
    result = model.invoke([HumanMessage(prompt)])
    return {
        "outline": result.content
    }

PROMPT_2 = (
    "根据以下内容⽣成⽂章完整初稿。\n"
    "主题：{topic}\n"
    "⼤纲: "
    "{outline}\n"
    "要求："
    "1.每个标题下，最多使⽤三句话的内容即可"
    "2.不⽤进⾏说明，只返回最终结果"
)
def generate_draft(state: OverallState):
    """编写初稿"""
    prompt = PROMPT_2.format(topic=state["topic"], outline=state["outline"])
    result = model.invoke([HumanMessage(prompt)])
    return {
        "draft": result.content
    }

def polish_content(state: OverallState):
    """内容润色"""
    pass

def finalize_content(state: OutputState):
    """最终整合"""
    pass