# 配置类
from pathlib import Path

from banks.filters import document

base_path = Path(__file__).parent
# 嵌入模型的路径
model_path = r"/Users/afeng./Data/RAG/afeng./model/cache/models/BAAI--bge-m3/snapshots/master"

document_path = base_path / "data"