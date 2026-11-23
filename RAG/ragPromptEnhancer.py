# 提示词增强
import shutil
import time
from typing import Dict, List

from llama_index.core import Document

import config
import milvus
import sentenceChunk

PROMPT_TEMPLATE = """
【角色】你是一个客服/技术顾问，只基于文档回答问题
【任务】根据文档回答用户的问题，不允许瞎编乱造
【上下文】
{context}
【用户的问题】
{question}
【输出要求】：
1. 准确，简洁，分条回答
2. 当用户查询无关信息的时候，直接回复“文档中不存在”
3. 不要用文档之外的知识回答
"""

DB_NAME = "ragPromptEnhancer"
COLLECTION = "ragPromptEnhancer"
DIMENSION = 1024

# 构建向量数据库
def init():
    # 判断DB是否存在
    db_path = config.base_path / f"{DB_NAME}.db"

    if db_path.exists():
        shutil.rmtree(db_path)
        time.sleep(0.5)

    # 创建DB
    milvus.create_db(DB_NAME)
    milvus.create_collection(COLLECTION, DIMENSION)

# 知识库入库
def index_to_milvus(doc_dir: str) -> int:
    # 句子分块
    chunks = sentenceChunk.sentence_chunk_document(doc_dir,max_sentences=1)
    if not chunks:
        raise ValueError(f"句子分块为空，请检查目录{doc_dir}")

    # 构造插入向量数据库的数据集
    rows = []
    for i, chunk in enumerate(chunks):
        # 计算向量
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append(
            {
                "id": i + 1,
                "vector": vec,
                "text": chunk.text,
            }
        )

    milvus.inert(COLLECTION, rows)
    print(f"已经完成入库{len(rows)}, 向量纬度：{DIMENSION}")
    return len(rows)

# 实现向量检索
def retrieve(query: str, tok_k: int=5) -> List[Dict]:
    milvus.get_client().load_collection(COLLECTION)
    results = milvus.search_by_text(
        COLLECTION,
        text=query,
        limit=tok_k,
        fields=["id", "text"],
    )

    return results[0] if results else []


# 上下文压缩
def compress_context(texts: List[str], max_chars: int=500) -> str:
    # 集合用于记录已经添加过的文本
    seen = set()
    parts = []
    total = 0 # 用来记录现在已经有了多少字符

    for text in texts:
        if text in seen:
            continue
        seen.add(text)

        sep_len = 2 if parts else 0   # 换行占位符

        # 对已有长度判断
        if total + sep_len + len(text) > max_chars:

            # 当前字符串容量
            remain = max_chars - total - sep_len

            if remain > 20:   # 20的容量是为了保证text语义的连贯性
                parts.append(text[:remain] + "...")
            break

        parts.append(text)
        total += sep_len + len(parts)

    return "\n\n".join(parts)

# 实现上下文的过滤
def filter_chunks(hits: List[Dict], max_distance: float=0.5) -> List[str]:
    texts = []

    for hit in hits:
        # distance是相似度
        if hit.get("distance") < max_distance:
            continue

        # entity:{"id","text","vector"} 字典
        text = hit.get("entity", {}).get("text", "").strip()

        # 假如在知识库里面查询出来的东西太短 那就直接丢掉
        if len(text) >= 10:
            texts.append(text)

    return texts

# 构造提示词
def build_prompt(question: str, context: str) -> str:
    if not context:
        context = "(无相关的文档片段)"
    return PROMPT_TEMPLATE.format(context=context, question=question)

# 根据原始文档和用户问题生成提示词
def get_prompt(
        question: str,
        top_k: int=5,
        max_distance: float=0.5,
        max_chars: int=500,
) -> str:
    # 检索出来的东西
    hits = retrieve(question, top_k)

    # 上下文的过滤
    texts = filter_chunks(hits, max_distance)

    # 上下文压缩
    context = compress_context(texts, max_chars)
    return build_prompt(question, context)


# 处理方法 循环生成增强后的提示词
def interactive(
        tok_k: int=5,
        max_distance: float=0.5,
        max_chars: int=500,
):
    while True:
        question = input("请输入用户的问题").strip()
        if question.lower() in "q":
            print("一推出循环")
            break
        if not question:
            continue

        prompt = get_prompt(question)
        print(prompt)

# 直接把分好块的进行入库
def index_to_milvus_by_chunks(chunks: List[Document])->int:

    # 定义待插入的数据集
    rows = []

    # 进行数据封装
    for i, chunk in enumerate(chunks):
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append({"id": i + 1, "vector": vec, "text": chunk.text})

    milvus.inert(COLLECTION, rows)
    print(f"已经完成入库：{len(rows)},向量的维度：{DIMENSION}")
    return len(rows)

if __name__ == "__main__":
    init()
    index_to_milvus(config.base_path / "data/1")
    interactive()
