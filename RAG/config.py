# 配置类
import os
from pathlib import Path
from dotenv import load_dotenv

base_path = Path(__file__).parent
# 嵌入模型的路径（随项目根目录走，换机器不用改）
model_path = str(base_path / "afeng." / "model" / "cache" / "models" / "BAAI--bge-m3" / "snapshots" / "master")

document_path = base_path / "data"

load_dotenv(base_path / ".env")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")


# 医疗问答项目全量
medical_all_data_path = base_path / "medical/data"

# 医疗问答项目简化版
medical_data_path = base_path / "medical/data/less"