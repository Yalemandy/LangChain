import re
from typing import List
from llama_index.core.schema import Document
import util
import config

CHUNK_DELIMITER = "===CHUNK==="

SYSTEM_PROMPT = f"""你是一个RAG知识库文档分块的助手，请把用户提供的文档切分成多个语义完整适合向量检索的分块
要求：
1.
每个分块围绕一个独立的主题
2. 尽量不要拆分段落，不要拆分句子
3. 保持原始文本的语义不变
4. 各个分块之间用{CHUNK_DELIMITER}分隔开
5. 只输出分块的正文，不要额外的信息
"""

def _parse_llm_chunks(response: str) ->List[str]:
    """
    解析LLM返回的文本
    :param response:
    :return:
    """
    parts = re.split(fr"\n?{CHUNK_DELIMITER}\n?", response.strip())
    chunks = [part.strip() for part in parts]

    if not chunks:
        chunks = [response.strip()]
    return chunks

def _llm_split_text(text: str, max_chunk_chars: int) -> List[str]:
    """
    调用LLM
    :param text:
    :param max_chunk_chars:
    :return:
    """
    USER_PROMPT = (
        f"请将以下文档分块，每个块不超过{max_chunk_chars}字符：\n\n"
        f"{text}"
    )

    all_chunks: List[str] = []

    result = util.chat(USER_PROMPT, SYSTEM_PROMPT)
    all_chunks.extend(_parse_llm_chunks(result))

    return all_chunks

def llm_chunk_documents(input_dir: str, max_chunk_chars: int=500) -> List[Document]:
    """
    调用LLM来分块
    :param input_str:
    :param max_chunk_chars:
    :return:
    """
    # 清洗文档
    documents = util.clean_all_formats(input_dir=input_dir)

    # 创建空chunk列表
    chunks: List[str] = []

    # 逐一执行LLM分块操作
    for d in documents:
        text = d.text

        # 调用LLM来处理分块
        text_chunks = _llm_split_text(text, max_chunk_chars)

        # 修改原数据
        for idx, chunk_text in enumerate(text_chunks):
            m = {"chunkindex":idx}
            chunks.append(Document(text=chunk_text, metadata=m))

    for i, chunk in enumerate(chunks):
        print(f"第{i + 1}个分块")
        print(chunk.metadata)
        print(chunk.text)
    return chunks

if __name__ == "__main__":
    llm_chunk_documents(config.base_path / "data/1")