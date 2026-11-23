# 数据分块
import threading
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config
import milvus
import ragPromptEnhancer
import util
from ragPromptEnhancer import DB_NAME, COLLECTION, DIMENSION


EMBED_BATH_SIZE = 32
INSERT_BATCH_SIZE = 200

_db_lock = threading.Lock()

def load_medical_qa(data_dir: str) -> List[Dict[str,str]]:
    # 读取目录下所有的cav文件，每行是一条回答
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob('*.csv')) # 收集csv文件
    if not csv_files:
        raise FileNotFoundError(f"当前目录{data_dir}里面没有csv文件")

    # 存解析后的问答列表
    qa_list: List[Dict[str,str]] = []
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    # 循环处理csv文件
    for csv_file in csv_files:
        df = None
        for enc in encodings:
            try :
                df = pd.read_csv(
                    csv_file,
                    encoding=enc,
                    names=["question", "answer"],
                    header=None,
                )
                break
            except UnicodeError:
                continue
        # 所有的编码格式打开csv文件都是失败的
        if df is None:
            df = pd.read_csv(
                csv_file,
                header=None,
                names=["question", "answer"],
            )
        df = df.fillna("")
        for _, row in df.iterrows():
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            if question:
                qa_list.append({
                    "question": question,
                    "answer": answer,
                })

    if not qa_list:
        raise ValueError(f"目录{data_dir} 中间没有有效的医疗问答数据")
    return qa_list

# 在进程内只创建一次milvus连接
def _ensure_milvus_connect() -> None:
    if milvus.get_client() is None:
        milvus.use_db(DB_NAME)

# 卸载collection的内存索引，还要保存milvus的连接不断开
def _release_collection_only():
    client = milvus.get_client()
    if client is None:
        return
    try:
        client.release_collection(COLLECTION)
    except Exception:
        pass

# 删除collection中的全部数据，为重建索引提供空间
def _delete_all_entities():
    client = milvus.get_client()
    try:
        client.delete(COLLECTION,filter="id>0")
        return
    except Exception:
        pass
    # 一次性删除一场，可以考虑分批删除
    bath_size = 1000
    while True:
        rows = client.query(
            COLLECTION,
            filter="id>0",
            output_fields=["id"],
            limit=bath_size
        )
        if not rows:
            break
        client.delete(COLLECTION, ids=[rows["id"] for row in rows])
        if len(rows) < bath_size:
            break

# 连接milbus数据库，吧collection加载到内存，供检索使用
def connect_db() -> None:
    _ensure_milvus_connect()
    client = milvus.get_client()
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)


# 创建向量数据库
def init_medical_db() -> None:
    # 获取milvus客户端
    _ensure_milvus_connect()
    client = milvus.get_client()

    # 首次建库
    if not client.has_collection(COLLECTION):
        client.create_collection(collection_name=COLLECTION, dimension=DIMENSION)
        return

    # 重建索引 先将内存中的集合卸载
    _release_collection_only()

    # 加载新的集合
    try:
        client.load_collection(COLLECTION)
    except Exception:
        pass

    # 清空旧的向量数据
    _delete_all_entities()
    _release_collection_only()

# 数据向量话
def build_milvus_rows(
        qa_list: List[Dict[str,str]],
        embed_bath_size: int=EMBED_BATH_SIZE,
) -> List[Dict[str,any]]:
    # 只转换question
    questions = [qa["question"] for qa in qa_list]
    # 记录所有向量数组
    vectors: List[List[float]] = []

    # 按照批次处理来向量化
    for start in range(0, len(questions), embed_bath_size):
        batch_text = questions[start:start+embed_bath_size]
        batch_vectors = milvus.embed_model.get_text_embedding_batch(texts=batch_text)
        vectors.extend(batch_vectors)
        # 看一下向量化的进度
        print(f"已经执行完向量化 {min(start + embed_bath_size, len(questions))}/{len(questions)}条")

    # 处理向量化
    if len(vectors) != len(qa_list):
        raise ValueError(f"向量数{len(vectors)}与问答数{len(qa_list)}数量不一致")

    # 构建待插入的数据
    rows: List[Dict[str,any]] = []
    for i, (qa, vector) in enumerate(zip(qa_list, vectors)):
        text = f"问：{qa['question']}\n答：{qa['answer']}"
        rows.append({
            "id": i,
            "vector": vector,
            "text": text,
            "question": qa["question"],
            "answer": qa["answer"],
        })
    print(f"构造待插入数据{len(rows)}条")
    return rows

#向量化的数据插入向量数据库
def insert_data(
        rows: List[Dict[str,any]],
        batch_size: int=INSERT_BATCH_SIZE,
) -> int:
    total = len(rows)
    inserted = 0

    for batch_idx, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start:start+batch_size]
        milvus.inert(COLLECTION, batch)
        inserted += len(batch)
        print(f"已经入库 {inserted}/{total}条", inserted, total)
        if start + batch_size < total:
            time.sleep(0.5)

        print(f"已经入库 {inserted}条, 向量维度{DIMENSION}")

# 5. 数据检索
def search_medical(question: str, top_k: int = 5) -> List[dict]:
    # 1. 连接数据库
    connect_db()
    # 2. 查询操作
    results = milvus.search_by_text(
        collection_name=COLLECTION,
        text=question,
        limit=top_k,
        fields=["id", "text", "question", "answer"],
    )
    # 3. 结果返回
    return results[0] if results else []

# 6. 提示词增强
def get_prompt(
        question:str,
        top_k:int=5,
        max_distance:float=0.5,
        max_chars:int=500,
) -> str:
    connect_db()
    return ragPromptEnhancer.get_prompt(
        question=question,
        top_k=top_k,
        max_distance=max_distance,
        max_chars=max_chars
    )

# 7. LLM生成
def get_answer(
        prompt: str,
) -> str:
    answer = "".join(util.stream_complete(prompt))
    return answer

# 8. 提供可被api调用的方法 -> 重建索引
def index_medical_to_milvus() -> int:
    with _db_lock:
        _release_collection_only()
        init_medical_db()
        qa_list = load_medical_qa(config.medical_data_path)
        rows = build_milvus_rows(qa_list)
        return insert_data(rows)

# 9. 提供可被api调用的方法   -> 用户提问
def query_medical(question: str) -> dict:
    with _db_lock:
        hits = search_medical(question=question)
        rank = 0
        sources =[]
        for hit in hits:
            text = hit.get("entity").get("text")
            rank +=1
            entity = hit.get("entity")
            sources.append(
                {
                    "rank":rank,
                    "score": round(1.0 - hit.get("distance", 1.0), 4),
                    "text":text,
                    "metadata":{
                        "question": entity.get("question"),
                        "id": entity.get("id"),
                    }
                }
            )
        prompt = get_prompt(question)
        _release_collection_only()
    answer = get_answer(prompt)
    if answer == "文档中不存在。":
        return {"answer":answer, "sources":""}
    return {"answer":answer, "sources":sources}

if __name__ == "__main__":
    # result: List[Dict[str,str]] = load_medical_qa(config.medical_data_path)
    # print(f"{len(result)}\n")
    # print(result)
    
    # result: List[Dict[str,str]] = load_medical_qa(config.medical_data_path)
    # rows = build_milvus_rows(result)
    # init_medical_db()
    # insert_data(rows)
    # results = search_medical(question="我有点头痛怎么办")
    # for result in results:
    #     print(result)

    prompt = get_prompt(question="我有点头痛怎么办")
    print(prompt)

    answer = get_answer(prompt)
    print("大模型返回的结果：")
    print(answer)