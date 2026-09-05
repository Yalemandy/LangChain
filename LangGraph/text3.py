import os
from typing import Literal

from langgraph.constants import START, END
from pydantic import BaseModel, Field

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, filter_messages
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# 1. LLM初始化
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ========== 加载文档 ==========
paths = [
    "D:\\Python\\LangChain/markdown/企业介绍.md",
    "D:\\Python\\LangChain/markdown/C++开发方向.md"
]

docs = []
for p in paths:
    loader = TextLoader(p, encoding="utf‑8")
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# 使⽤内存中向量存储和 OpenAI 嵌⼊
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
# 使⽤ LangChain 的预构建 create_retriever_tool 创建检索器⼯具：
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_bit",
    "搜索并返回有关⽐特就业课的信息。",
)

# 1. 定义节点
def generate_query_or_respond(state: MessagesState):
    """用于调用模型以基于当前状态生成响应
    给定问题，她讲决定使用工具检索，或者简单的生成用户响应"""

    result = model.bind_tools([retriever_tool]).invoke(state["messages"])

    return {
        "messages": [result]
    }

# generate_query_or_respond({
#     "messages": [
#         {
#             "role": "user",
#             "content": "C++开发方向需要掌握哪些技能？"
#         }
#     ]
# })["messages"][-1].pretty_print()

# 节点2
# 检索器工具 ToolNode是内置的节点，用于调用工具
retrieve_node = ToolNode([retriever_tool])

# 节点3
REWRITE_PROMPT = (
    "查看输⼊并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出⼀个改进后的问题："
)
def rewrite_question(state: MessagesState):
    """该节点用来重写问题是否符合模型要求"""

    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    result = model.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [HumanMessage(content=result.content)]
    }

GENERATE_PROMPT = (
    "你是负责回答问题的助⼿。 "
    "使⽤以下检索到的上下⽂⽚段来回答问题。 "
    "如果你不知道答案，就说你不知道。 "
    "最多只⽤三句话，回答要简明扼要。\n"
    "Question: {question} \n"
    "Context: {context}"
)
def generate_answer(state: MessagesState):
    """生成答案"""

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question,context=context)
    result = model.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [result]
    }

# 构建图 添加节点 添加边
builder = StateGraph(MessagesState)

builder.add_node(generate_query_or_respond)
builder.add_node("retrieve", retrieve_node)
builder.add_node(rewrite_question)
builder.add_node(generate_answer)

builder.add_edge(START,"generate_query_or_respond")
builder.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # 判断最后一个AIMessage是否使用了工具
    {
        "tools": "retrieve",
        "__end__": END
    }
)
GRADE_PROMPT = (
    "你是⼀个评分员，评估检索到的⽂档与⽤⼾问题的相关性。 \n "
    "以下是检索到的⽂档： \n\n {context} \n\n"
    "以下是⽤⼾的问题： {question} \n"
    "如果⽂档包含与⽤⼾问题相关的关键字或语义，则将其评为相关。 \n"
    "给出⼀个⼆元分数“yes”或“no”，以表明该⽂档是否与问题相关。\n"
    "请以json格式返回，必须包含score字段，例如：{{\"score\": \"yes\"}} 或 {{\"score\": \"no\"}}"
)
class GradeDocuments(BaseModel):
    score: str = Field(description="相关性评分，如果文档与问题相关，则评分为'yes'，否则评分为'no'")

def grade_docements(state: MessagesState) -> Literal["rewrite_question", "generate_answer"]:
    """调用llm来判断最后一个答复是否符合预期"""

    user_message = filter_messages(state["messages"],include_types="human")
    question = user_message[-1].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = model.with_structured_output(GradeDocuments, method="json_mode").invoke([HumanMessage(content=prompt)])

    if result.score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


builder.add_conditional_edges(
    "retrieve",
    grade_docements,
    ["rewrite_question", "generate_answer"]
)
builder.add_edge("generate_answer",END)
builder.add_edge("rewrite_question","generate_query_or_respond")

# 编译图
graph = builder.compile()

# 执行图
for check in graph.stream(
    {
        "messages": [HumanMessage(content="Java方向有哪些课程？")]
    }
):
    # print(check)
    for node, updata in check.items():
        print(f"由节点{node}更新信息")
        updata["messages"][-1].pretty_print()
        print("\n\n")