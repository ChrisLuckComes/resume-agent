from fastapi import (
    FastAPI,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
)
from sqlalchemy import select
from models import QueryRequest, EvaluationRequest, Resume, ResumeStatus
from agent import ResumeAgent
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from database import engine, get_db, redis_client, Base
import os
import shutil


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接数据库
    # 1. 【启动时】初始化表结构 (替代原来的 agent.init_db)
    async with engine.begin() as conn:
        # 这行代码会检查 models.py 中定义的 Base 类，并创建所有不存在的表
        await conn.run_sync(Base.metadata.create_all)

    print("数据库连接已建立，表结构已初始化")
    yield
    # 关闭时断开数据库连接
    await agent.engine.dispose()
    await agent.redis_client.close()
    print("数据库和Redis连接已断开")


app = FastAPI(lifespan=lifespan)
agent = ResumeAgent(redis_client)


@app.get("/")
async def root():
    return {"message": "AI职业经纪人已启动，访问/chat接口进行对话"}


@app.post(
    "/upload_resume",
    summary="上传简历文件",
    description="上传docx格式的简历文件，AI经纪人将解析内容并存储",
)
async def upload_resume(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    phone: str = Form(...),
    user_id: str = Form(...),
):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="仅支持docx格式的文件")

    # 先查询是否存在记录
    stmt = select(Resume).where(Resume.user_id == user_id, Resume.phone == phone)
    result = await db.execute(stmt)
    existing_resume = result.scalar_one_or_none()

    if existing_resume:
        # 如果记录存在，更新状态为PENDING，等待重新处理
        existing_resume.status = ResumeStatus.PENDING
        new_resume = existing_resume
    else:
        # 数据库创建初始记录
        new_resume = Resume(
            user_id=user_id,
            candidate_name=candidate_name,
            phone=phone,
            status=ResumeStatus.PENDING,  # 初始状态为解析中
        )
        db.add(new_resume)
    await db.commit()
    await db.refresh(new_resume)

    background_tasks.add_task(
        agent.handle_resume_process,
        resume_id=new_resume.id,
        file_content=await file.read(),
        candidate_name=candidate_name,
        phone=phone,
    )

    return {
        "message": "简历上传成功，正在后台处理",
        "resume_id": new_resume.id,
        "status": "pending",
    }

    # # 1. 临时保存文件
    # temp_path = f"temp_{file.filename}"
    # with open(temp_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    # try:
    #     # 2. 调用Agent方法处理文件
    #     num_chunks = await agent.add_resume(temp_path, candidate_name, phone)
    #     return {
    #         "status": "success",
    #         "candidate_name": candidate_name,
    #         "chunks_added": num_chunks,
    #         "message": "简历已解析并存入向量库",
    #     }
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     # 3. 删除临时文件
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)


@app.post(
    "/evaluate",
    summary="评估简历",
    description="根据候选人姓名和简历内容，AI经纪人返回评估结果",
)
async def evaluate_resume(request: EvaluationRequest, db: Session = Depends(get_db)):
    # 1. 先查询数据库有没有这个简历
    existing_resume = await agent.check_and_get_evaluation(
        request.user_id, request.candidate_name, request.phone, db=db
    )

    if not existing_resume:
        # 2. 如果没有，就返回一个提示，要求先上传简历
        raise HTTPException(
            status_code=404,
            detail="未找到对应的简历，请先上传简历后再进行评估",
        )

    if existing_resume.evaluation_result:
        return {"evaluation": existing_resume.evaluation_result}

    evaluation = await agent.evaluate_resume(existing_resume.content)
    return {"evaluation": evaluation}


@app.post("/query", summary="查询接口", description="发送查询文本，AI经纪人返回回答")
async def query_resume(
    question: str = Form(...),
    user_id: str = Form(...),
    candidate_name: str = Form(None),
    db: Session = Depends(get_db),
):
    # 构建过滤条件
    search_filter = None
    if candidate_name:
        search_filter = {"candidate": candidate_name}
    answer = await agent.ask(question, user_id, search_filter, db=db)
    return {"reply": answer}


@app.post(
    "/chat",
    summary="与AI职业经纪人对话",
    description="发送问题给 AI 经纪人，支持多轮对话记忆",
)
async def chat_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    answer = await agent.ask(request.text, request.user_id, db=db)
    background_tasks.add_task(agent.async_persistence, request.user_id, answer)
    return {"reply": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
