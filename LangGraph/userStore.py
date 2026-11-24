import uuid

from langgraph.store.postgres import PostgresStore
# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

import config

embeddings = HuggingFaceEmbeddings(
    model_name=config.model_path,
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"local_files_only": True}
)

# 1. 内存级 store
store = InMemoryStore(
    index={
        "embed": embeddings,   # 模型名
        "dims": 1024,          # 维度
        "fields": ["$"]        # 对value中的所有字段进行嵌入
    }
)

# store 基本使用
# namespace的命名，用元组来做， namespace1 = ("","","")   prefs 偏好

# 2. 定义命名空间   用于组织记忆，通常按照业务来进行划分，例如：定制化智能助手需要用用户id进行隔离
user_id = "user_1"
namespace1 = (user_id, "prefs","food")
namespace2 = (user_id, "prefs", "music")
#
# # 3. 存进记忆
memory_id1 = str(uuid.uuid4())
memory_value1 = {"china": "披萨"}

memory_id2 = str(uuid.uuid4())
memory_value2 = {"china": "雨夜"}

# # 4. 记忆存储
store.put(
    namespace=namespace1,
    value=memory_value1,
    key=memory_id1,
)
store.put(
    namespace=namespace2,
    value=memory_value2,
    key=memory_id2,
)

# 5. 记忆读取
# print(store.get(namespace1, memory_id1))
# all = store.search((user_id, "prefs"))
# print(all)
# for item in all:
#     print(item.dict())

# query可以按照语义来进行搜索 store 支持语义查询
all = store.search((user_id, "prefs"), query="用户喜欢的中文音乐是什么")
print(all)

DB_URL = "postgresql://postgres:yalemandy@127.0.0.1:5432/langgraph"
with(
    PostgresStore.from_conn_string(DB_URL) as checkpointer,
    PostgresStore.from_conn_string(DB_URL) as store,
):
    # 第一次setup
    store.setup()
    checkpointer.setup()
    user_id = "user_1"
    namespace = (user_id, "prefs", "food")
    memory_id = "d12546fb-2b61-4053-a44e-79b345454d9b"
    memory_value = {"china": "披萨"}
    store.put(
        namespace=namespace,
        value=memory_value,
        key=memory_id,
    )

    print(store.get(
        namespace=namespace,
        key=memory_id,
    ))
