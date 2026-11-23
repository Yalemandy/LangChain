import re
from pathlib import Path
from typing import List

from llama_index.core import Document

from config import base_path
from util import parse_all_formats

# 1. 直接操作一个目录
# 2. 一个关键参数 几个句子分一块
# 3. 接受和返回：Document列表

def sentence_chunk_document(input_str: str, max_sentences: int =5)->List[Document]:
    """
    根据句子分块
    :param input_str:
    :param max_sentences:
    :return:
    """
    # 准备一个可以接受结果的列表
    chunks = []
    # 清洗数据
    documents = parse_all_formats(input_dir=input_str)
    # 遍历每个document对象，进行句子分块的操作
    for doc in documents:
        text = doc.text
        path = doc.metadata["file_path"]

        sents = [
            s.strip()
            for s in re.split(r"(?<=[。！？！？\.])\s+", text)
            if s.strip()
        ]

        for i in range(0, len(sents), max_sentences):
            slice_sents = sents[i:i + max_sentences]
            valid_sents = [s for s in slice_sents if isinstance(s, str)]
            part = " ".join(valid_sents).strip()
            if not part:
                continue

            m = dict(doc.metadata)
            m.update({
                "source_file_path": path,
                "chunk_start": i,
                "chunk_index": i,
                "chunk_end": min(i+max_sentences, len(sents)),
            })
            chunks.append(Document(text=part, metadata=m))

    # 预览效果
    for i, chunk in enumerate(chunks):
        print(f"第{i + 1}个分块")
        print(chunk.metadata)
        print(chunk.text)

    return chunks

if __name__ == "__main__":
    sentence_chunk_document(Path(base_path) / "data")