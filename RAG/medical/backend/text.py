from fastapi import FastAPI
import uvicorn
# 创建一个FastAPI实例
app = FastAPI()

@app.get("/hello")
def hello():
    return {"msg": "Hello World"}



if __name__ == "__main__":
    uvicorn.run(app="text:app", host="127.0.0.1", port=8080)