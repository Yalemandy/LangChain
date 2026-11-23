import shutil
import time

from markdown_it.rules_inline import entity

import config
import milvus
from sentenceChunk import sentence_chunk_document

# 向量检索


# 构建向量数据库

DB_NAME = "vectorSearch"
COLLECTION = "vectorSearch"
DIMENSION = 1024

def init():
    # 判断DB是否存在
    db_path = config.base_path / f"{DB_NAME}.db"

    if db_path.exists():
        shutil.rmtree(db_path)
        time.sleep(0.5)

    # 创建DB
    milvus.create_db(DB_NAME)
    milvus.create_collection(COLLECTION, DIMENSION)

# 将文本分块，分块内容逐步向量化入库 句子分块
def index_to_milvus():
    chunks = sentence_chunk_document(config.base_path / "data/1", 1)
    if not chunks:
        raise ValueError("分块唯恐，检查数据集内容")

    rows = []

    # 将分好块的内容入库
    for i, chunk in enumerate(chunks):
        vec = milvus.embed_model.get_text_embedding(chunk.text)
        rows.append(
            {
                "id": i + 1,
                "vector": vec,
                "text": chunk.text,
            }
        )

    milvus.inert(COLLECTION, rows)
    print(f"已完成入库，{len(rows)}个记录，向量纬度：{DIMENSION}")
    return rows

# 文本检索
def search_similar(query: str, limit: int = 2):
    milvus.get_client().load_collection(COLLECTION)
    results = milvus.search_by_text(
        COLLECTION,
        text=query,
        limit=limit,
        fields=["id","vector","text"]
    )

    hits = results[0] if results else []
    if not hits:
        print("为检索到相关内容")
        return
    print(f"\n用户查询{query}")
    print(f"共{len(hits)}条相似的结果")
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity")
        text = entity.get("text")
        print(f"\n{rank} 距离 distance={hit.get('distance'):.4f}")
        print(f"id={entity.get('id',hit.get('id'))}")
        print(f"{text}\n")

# 循环输入用户查询，获取知识库查询
def search():
    print("\n向量库已经准备就绪，请输入用户问题：")
    while True:
        query = input("请输入问题：").strip()
        if query.lower() in ("q"):
            print("已经推出!")
            break
        if not query:
            continue
        search_similar(query)


if __name__ == "__main__":
    init()
    index_to_milvus()
    search()
