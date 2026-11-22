from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config


# 计算两段文本相似度
def embedding_similarity(sentence1: str , sentence2: str) -> float:

    # 模型加载（model_name 需要字符串，config.model_path 是 Path 对象）
    embed_model = HuggingFaceEmbedding(model_name=str(config.model_path))

    # 计算文本向量
    vec1 = embed_model.get_text_embedding(sentence1)
    vec2 = embed_model.get_text_embedding(sentence2)

    # 计算相似度
    similarity = embed_model.similarity(vec1, vec2)

    return similarity

if __name__ == "__main__":
    print(embedding_similarity("小米真帅", "小米真丑"))