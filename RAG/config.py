# 配置类
from pathlib import Path

base_path = Path(__file__).parent
# 嵌入模型的路径（随项目根目录走，换机器不用改）
model_path = base_path / "afeng." / "model" / "cache" / "models" / "BAAI--bge-m3" / "snapshots" / "master"

document_path = base_path / "data"