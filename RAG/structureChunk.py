from typing import List
from llama_index.core import Document

import util
import config

max_chunk_chars = 1500

def _extract_section_text(section: dict) -> str:
    """
    对ppt的解析结果提取文本
    :param section:
    :return:
    """
    return section.get("content")


def chunk_ppt_document(d: Document) -> List[Document]:
    """
    对ppt进行分块
    :param d:
    :return:
    """
    sections = d.metadata.get("text_sections")
    slide_title =d.metadata.get("title")

    texts = [_extract_section_text(s) for s in sections]
    texts = [t for t in texts if t]

    full_text = "\n".join(texts)
    if len(full_text) <= max_chunk_chars:
        meta = {
            "chunk_text": 0
        }
        return [Document(text=full_text, metadata=meta)]

    groups = []
    current = []

    for i, text in enumerate(texts):
        current.append(text)

    if current:
        groups.append("\n".join(current))

    # 转换成分块
    chunks: List[Document] = []
    for idx, group in enumerate(groups):
        chunk_text = group
        if slide_title and slide_title not in chunk_text:
            chunk_text = f"{slide_title}\n{chunk_text}"

        meta = {
            "section_index": idx,
        }
        chunks.append(Document(text=chunk_text, metadata=meta))

    return chunks


def chunk_single_document(d: Document) -> List[Document]:
    """
    按照文档类型来分块
    :param d:
    :return:
    """
    # 先来处理ppt（解析器输出的键名是 text_sections，每页一个 Document）
    if d.metadata.get("text_sections"):
        return chunk_ppt_document(d)

    # 其他类型（md/csv/pdf 等）的分块逻辑尚未实现，先跳过，避免返回 None
    return []

def structure_chunk_document(input_str: str) -> List[Document]:
    """

    :param input_str:
    :return:
    """

    # 清洗文档
    documents = util.parse_all_formats(input_str)

    # 逐一分块
    chunks = []
    for doc in documents:
        doc_chunks = chunk_single_document(doc)
        chunks.extend(doc_chunks)

    # 测试输出
    for i, chunk in enumerate(chunks):
        print(f"第{i+1}个分块")
        print(chunk.metadata)
        print(chunk.text)

    return chunks

if __name__ == "__main__":
    structure_chunk_document(config.document_path)