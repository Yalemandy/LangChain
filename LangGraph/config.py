import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

base_path = Path.home() / "Data" / "LangChain" / "RAG"

# 嵌入模型的路径（随项目根目录走，换机器不用改）
model_path = str(base_path / "afeng." / "model" / "cache" / "models" / "BAAI--bge-m3" / "snapshots" / "master")

load_dotenv(base_path / ".env")
api_key = os.getenv("DEEPSEEK_API_KEY")


model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=api_key,
    temperature=0
)