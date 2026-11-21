from typing import List
from llama_index.core import Document
from llama_index.core.async_utils import chunks
from llama_index.core.node_parser import TokenTextSplitter
from scipy.sparse.linalg import spilu

import util
import config

# 递归分块
def recuersize_chunk_docuemnts(input_str: str) -> List[Document]:
    """
    对指定文档的递归分块
    :param input_str: 指定路径
    :return: 分块好的列表
    """
    # 清洗后的数据
    documents = util.parse_all_formats(input_str)
    chunks = []

    # 根据符号划分文本
    # 分 隔 符 优 先级 列 表 ：从 前 到后 依 次 尝 试 ，越靠 前 越优 先 （语 义 边 界 越完 整 ）
    # 段 落 -> 换行 -> 中 英文句末 -> 分 号 -> 逗 号 -> 空 格 -> 单字符
    separators = [
        "\n\n","\n",
        "。","！"
        "？",""
        "；",";"
        "，",","
        " ",""
    ]

    # 创建分块器
    spliter = TokenTextSplitter(
        separator=separators[0],
        backup_separators=separators[1:],
    )

    # 使用分块器来处理清洗好的文档
    for document in documents:
        text = document.text
        if not text:
            continue

        path = document.metadata.get("file_path")

        nodes = spliter.get_nodes_from_documents(
            [Document(text=text,metadata={"file_path": path})],
        )

        for idx, n in enumerate(nodes):
            m = dict(n.metadata)
            m.update({
                "source_file_path": path,
                "chunk_index": idx
            })
            chunks.append(Document(text=text,metadata=m))

    for i, chunk in enumerate(chunks):
        print(f"第{i + 1}个分块")
        print(chunk.metadata)
        print(chunk.text + f"\n")

    return chunks

if __name__ == "__main__":
    recuersize_chunk_docuemnts(config.document_path)
