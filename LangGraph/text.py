import warnings

# 屏蔽langchain‑community弃用警告
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset"
)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# -------- 1.加载md文档 --------
file_paths = [
    "D:/Python/LangChain/markdown/C++开发方向.md",
    "D:/Python/LangChain/markdown/企业介绍.md",
]
docs = []
for fp in file_paths:
    loader = TextLoader(fp, encoding="utf-8")
    docs.extend(loader.load())

# -------- 2.文本切分 --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
split_docs = splitter.split_documents(docs)

# -------- 3.本地中文向量模型，不走远程API --------
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# -------- 4.内存向量库 --------
vectorstore = InMemoryVectorStore.from_documents(
    documents=split_docs,
    embedding=embedding
)
retriever = vectorstore.as_retriever(k=3)

# 简单测试检索
res = retriever.invoke("测试问题")
print("检索结果：")
for doc in res:
    print(doc.page_content[:200])
