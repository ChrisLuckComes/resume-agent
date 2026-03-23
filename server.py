from fastapi import FastAPI
from pydantic import BaseModel
from agent import ResumeAgent
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接数据库
    await agent.init_db()
    yield
    # 关闭时断开数据库连接
    await agent.engine.dispose()
    await agent.redis_client.close()

app = FastAPI(lifespan=lifespan)
agent = ResumeAgent()


class QueryRequest(BaseModel):
    user_id: str # 标识用户
    text: str



@app.get("/")
async def root():
    return {"message": "AI职业经纪人已启动，访问/chat接口进行对话"}


@app.post(
    "/chat",
    summary="与AI职业经纪人对话",
    description="发送问题给 AI 经纪人，支持多轮对话记忆",
)
async def chat_endpoint(request: QueryRequest):
    answer = await agent.ask(request.text, request.user_id)
    return {"reply": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
