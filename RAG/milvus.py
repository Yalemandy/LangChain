import sqlite3
from typing import List, Dict, Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymilvus import MilvusClient

import config

client: MilvusClient = None

# 连接milvus
def connect(dbpath: str) -> MilvusClient:
    global client
    client = MilvusClient(str(dbpath))

    return client

# 获取client
def get_client() -> MilvusClient:
    return client

# 创建DB
def create_db(dbname: str) -> MilvusClient:
    return connect(config.base_path / f"{dbname}.db")


# use db 切换DB
def use_db(dbname: str) -> MilvusClient:
    return connect(config.base_path / f"{dbname}.db")

# 列出所有的DB
def list_db() -> List[str]:
    return [p.stem for p in config.base_path.glob("*.db")]

# 创建collection
def create_collection(name: str, dimension: int) -> None:
    client = get_client()
    if client.has_collection(name):
        client.drop_collection(collection_name=name)

    client.create_collection(collection_name=name,dimension=dimension)

# 删除collection
def drop_collection(name: str) -> None:
    get_client().drop_collection(collection_name=name)

# 增删改查
def inert(collection_name: str, rows: List[Dict[str,Any]]) -> Dict:
    return get_client().insert(collection_name=collection_name, data=rows)

def get_by_id(collection_name: str, ids: List[int], fileds: Optional[List[str]] = None) -> List[Dict]:
    return get_client().get(collection_name=collection_name, ids=ids, output_fields=fileds)

embed_model = HuggingFaceEmbedding(model_name=config.model_path)

# 向量查询
def search_by_text(collection_name: str, limit: int=5, text: str="", fields: Optional[List[str]] = None)\
        -> List[List[Dict]]:
    vecors = [embed_model.get_text_embedding(text)]
    return get_client().search(
        collection_name=collection_name,
        limit=limit,
        data=vecors,
        output_fields=fields,
    )


if __name__ == "__main__":
    # create_db("yalemandy")
    use_db("yalemandy")
    # create_collection("01",1024)
    # drop_collection("01")
    # embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    # text = "我是小丹丹"
    # vec = embed_model.get_text_embedding(text)
    #
    # print(inert("01", [{"id": 1, "vector": vec, "text": text}]))
    # print(inert("01", [{"id": 2, "vector": vec, "text": text}]))
    # print(inert("01", [{"id": 3, "vector": vec, "text": text}]))
    # print(inert("01", [{"id": 4, "vector": vec, "text": text}]))

    # 现加载内存里，从磁盘
    get_client().load_collection("01")

    print(get_by_id("01", ids=[1]))