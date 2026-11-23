from aiohttp.abc import HTTPException
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from medical.backend.main import index_medical_to_milvus, query_medical

# 创建实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允许所有来源的跨域请求
    allow_headers=["*"],   # 允许所有的请求头
    allow_methods=["*"],      # 允许所有的http方法
)

@app.post("/api/rebuild-index")
def rebuild_index():
    try:
        count = index_medical_to_milvus()
        return {"ok": True, "message": f"索引构建完成，写入milvus{count}条数据"}
    except Exception as e:
        print(f"构建索引失败{e}")
        raise HTTPException(status_code=500, detail="索引构建失败，请重试")

class AskRequest(BaseModel):
    question: str

# 接收用户提问并回答
@app.post("/api/ask")
def ask_question(payload : AskRequest) -> dict:
    # 接收用户的提问
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return query_medical(q)
    except Exception as e:
        print(f"查询失败{e}")
        raise HTTPException(status_code=500, detail="请检查一下向量数据库是否构建成功")

if __name__ == "__main__":
    uvicorn.run(app="api:app", host="127.0.0.1", port=8080)
