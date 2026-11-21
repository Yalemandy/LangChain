# 固定大小分块
from pathlib import Path
from typing import List

from config import base_path
from util import clean_all_formats
from llama_index.core import Document


# 1. 直接操作一个目录
# 2. 两个关键参数，一个是块大小，一个是重叠区大小
# 3. 接受和返回：Document列表

def fixed_size_chunk_decuments(input_dir:str, chunk_size: int=500, chunk_overlap: int=50)-> List[Document]:
    """

    :param input_dir: 文档路径
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠长度
    :return: 一个按照固定大小切好块的document列表
    """
    # 滑动步长，准备一个可以接受结果的列表
    step = chunk_size - chunk_overlap
    chunks = []

    # 清洗文档
    docus = clean_all_formats(input_dir)
    docu_index = 0

    # 遍历Document对象，进行分块
    for doc in docus:
        text = doc.text
        path = doc.metadata.get("file_path")
        start = 0

    # while便利进行滑动窗口分块
        while start < len(text):
            end = min(len(text), start + chunk_size)
            # 考虑预览效果
            m = dict(doc.metadata)
            m.update({
                "path": path,
                "chunk_start": start,
                "chunk_end": end,
                "document_index": docu_index,
            })
            chunks.append(Document(text=text[start:end],metadata=m))

            if end >= len(text):
                break

            start = start + step
        docu_index = docu_index + 1

    # 预览
    for i, d in enumerate(chunks):
        print(f"第{i + 1}个分块")
        metadata = d.metadata
        print(f"文件路径：{metadata.get('file_path')} | "
              f"分块的起始位置：{metadata.get('chunk_start')} "
              f"| 分块的终止位置：{metadata.get('chunk_end')}| "
              f"分块前的索引位置：{metadata.get('document_index')}")
        print(d.text)
        print("-" * 40)

    return chunks

if __name__ == "__main__":
    fixed_size_chunk_decuments(Path(base_path) / "data")