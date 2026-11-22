# 语义分块
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
import util


# 先考虑加载嵌入模型
# 如何吧文本转成向量
# 计算向量相似度
# 打印中间的结果

def text_embedding_similarity(sentence1: str, sentence2: str) -> float:
    """
    比较两个向量的相似度
    :param sentence1:
    :param sentence2:
    :return:
    """
    # 加载嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=config.model_path, device="mps")

    # 如何吧文本转成向量
    vec1 = embed_model.get_text_embedding(sentence1)
    vec2 = embed_model.get_text_embedding(sentence2)

    # 计算向量相似度
    similarity = embed_model.similarity(vec1, vec2)

    # 打印中间的结果
    print(f"\n句子1：{sentence1}")
    print(f"\n向量1： （维度 {len(vec1)}）: {vec1}")
    print(f"\n句子2：{sentence2}")
    print(f"\n向量2： （维度 {len(vec2)}）: {vec2}")
    print(f"\n句子相似度：{similarity:.4f}")

    return similarity

# 1. 直接操作一个目录
# 2. 接受和返回：Document列表
def semantic_chunk_document(input_str: str) -> List[Document]:
    """

    :param input_str: 目标文件路径
    :return: 切好的document列表
    """

    # 获取清洗之后的Document列表 空列表
    documents = util.clean_all_formats(input_str)
    chunks = []
    # 加载嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=config.model_path)
    # 语义分割节点的解析器对象
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=10,
        breakpoint_percentile_threshold=95
    )
    # 执行语义分块操作
    nodes = splitter.get_nodes_from_documents(
        documents=documents,
    )

    for i, node in enumerate(nodes):
        print(f"第{i + 1}个分块")
        print(node.metadata)
        print(node.text)

    # 将得到的结果返回给到空列表里面
    for node in nodes:
        chunks.append(Document(text=node.text,metadata=node.metadata))

    return chunks


if __name__ == "__main__":
    # text_embedding_similarity("小明是个大帅哥", "小明长的好帅")
    # text_embedding_similarity("小明长的好帅", "小明长的好丑")
    semantic_chunk_document(config.document_path)