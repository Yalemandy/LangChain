# 数据集->数据加载->文本分块(各种分块都用下)->嵌入模型->向量入库->数据检索->提示词增强->LLM生成
from typing import List

from llama_index.core import Document

import config
import fixedSizeChunk
import ragPromptEnhancer
import sentenceChunk
import util

if __name__ == "__main__":
    # 生成数据库
    ragPromptEnhancer.init()
    # 数据加载
    chunks: List[Document] = []

    # chunks = fixedSizeChunk.fixed_size_chunk_decuments(config.base_path / "data/1")  # 固定分块
    chunks = sentenceChunk.sentence_chunk_document(config.base_path / "data/1")

    # 数据入库
    ragPromptEnhancer.index_to_milvus_by_chunks(chunks)

    # 提示词增强
    while True:
        question = input("请输入问题：").strip()
        if question.lower() in ("q"):
            print("已经推出!")
            break
        if not question:
            print("question无效")
            continue
        prompt = ragPromptEnhancer.get_prompt(question)
        print(prompt)
        print("大模型生成的内容：\n")

        for text in util.stream_complete(prompt):
            print(text, end="", flush=True)




