import hashlib
import inspect
import json

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import ResumeAgent
from database import Base, engine, get_db, redis_client
from models import ChatRequest, EvaluationRequest, JDAnalysisRequest, QueryRequest, Resume, ResumeStatus


async def _close_redis() -> None:
    if not redis_client:
        return
    close_method = getattr(redis_client, "aclose", None) or getattr(redis_client, "close", None)
    if close_method is None:
        return
    result = close_method()
    if inspect.isawaitable(result):
        await result


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    await _close_redis()


app = FastAPI(lifespan=lifespan)
agent = ResumeAgent(redis_client)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "resume-agent backend is running",
        "endpoints": [
            "/analyze_jd",
            "/ocr_jd_image",
            "/upload_resume",
            "/resumes",
            "/evaluate",
            "/chat",
        ],
    }


@app.post("/analyze_jd", summary="分析JD并提取关键词")
async def analyze_jd(request: JDAnalysisRequest):
    jd_text = request.jd_text.strip()
    if not jd_text:
        return {"keywords": []}

    cache_payload = f"{request.target_seniority or ''}\n{jd_text}"
    cache_key = f"jd_analysis:{hashlib.md5(cache_payload.encode('utf-8')).hexdigest()}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    result = (await agent.analyze_jd(jd_text, request.target_seniority)).model_dump()
    if redis_client:
        await redis_client.setex(
            cache_key,
            3600,
            json.dumps(result, ensure_ascii=False),
        )
    return result


@app.post("/ocr_jd_image", summary="图片OCR提取JD文字")
async def ocr_jd_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件进行OCR")

    extracted = await agent.extract_text_from_image(
        file_bytes=await file.read(),
        mime_type=file.content_type,
    )
    return extracted.model_dump()


@app.post("/upload_resume", summary="上传并解析简历")
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    phone: str = Form(...),
    user_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="仅支持 .docx 或 .pdf 格式的文件")

    stmt = select(Resume).where(Resume.user_id == user_id, Resume.phone == phone)
    existing_resume = (await db.execute(stmt)).scalar_one_or_none()

    if existing_resume:
        resume = existing_resume
        resume.candidate_name = candidate_name
        resume.status = ResumeStatus.PENDING
        resume.content = None
        resume.evaluation_result = None
    else:
        resume = Resume(
            user_id=user_id,
            candidate_name=candidate_name,
            phone=phone,
            status=ResumeStatus.PENDING,
        )
        db.add(resume)

    await db.flush()

    try:
        resume.status = ResumeStatus.PARSING
        raw_text = await agent.ingest_resume(
            file_name=file.filename,
            file_content=await file.read(),
            user_id=user_id,
            resume_id=resume.id,
            candidate_name=candidate_name,
            phone=phone,
        )
        resume.content = raw_text
        resume.status = ResumeStatus.COMPLETED
        await db.flush()
    except Exception as exc:
        resume.status = ResumeStatus.FAILED
        await db.flush()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "简历上传并解析完成",
        "resume_id": resume.id,
        "status": resume.status.value,
    }


@app.get("/resumes", summary="查询已上传简历列表")
async def list_resumes(user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Resume).where(Resume.user_id == user_id).order_by(desc(Resume.updated_at))
    resumes = (await db.execute(stmt)).scalars().all()
    return {
        "items": [
            {
                "resume_id": resume.id,
                "candidate_name": resume.candidate_name,
                "phone": resume.phone,
                "status": resume.status.value,
                "updated_at": resume.updated_at.isoformat(),
            }
            for resume in resumes
        ]
    }


@app.delete("/resumes/{resume_id}", summary="删除已上传简历")
async def delete_resume(resume_id: int, user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    resume = (await db.execute(stmt)).scalar_one_or_none()
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")

    agent.delete_resume_vectors(
        user_id=user_id,
        resume_id=resume.id,
        candidate_name=resume.candidate_name,
    )
    await db.delete(resume)
    await db.flush()
    return {"message": "简历已删除", "resume_id": resume_id}


@app.post("/evaluate", summary="评估简历")
async def evaluate_resume(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")

    resume = await _find_resume(request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")

    resume.status = ResumeStatus.EVALUATING
    await db.flush()

    evaluation = await agent.evaluate_resume(
        resume_text=resume.content,
        jd_text=jd_text,
        jd_keywords=request.jd_keywords,
        target_seniority=request.target_seniority,
    )
    resume.evaluation_result = evaluation
    resume.status = ResumeStatus.COMPLETED
    await db.flush()
    return {"evaluation": evaluation}


@app.post("/query", summary="非流式查询")
async def query_resume(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    reply = await agent.ask(
        question=request.text,
        user_id=request.user_id,
        db=db,
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
    )
    return {"reply": reply}


@app.post("/chat", summary="流式聊天")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    async def event_stream():
        full_response_parts: list[str] = []
        try:
            async for chunk in agent.stream_chat(
                question=payload.text,
                user_id=payload.user_id,
                db=db,
                candidate_name=payload.candidate_name,
                resume_id=payload.resume_id,
            ):
                if not chunk:
                    continue
                full_response_parts.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            suggestions = await agent.generate_follow_up_suggestions(
                question=payload.text,
                answer="".join(full_response_parts),
                candidate_name=payload.candidate_name,
            )
            yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _find_resume(request: EvaluationRequest, db: AsyncSession) -> Resume | None:
    if request.resume_id is not None:
        stmt = select(Resume).where(
            Resume.id == request.resume_id,
            Resume.user_id == request.user_id,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    if request.phone:
        stmt = select(Resume).where(
            Resume.user_id == request.user_id,
            Resume.phone == request.phone,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    if request.candidate_name:
        stmt = select(Resume).where(
            Resume.user_id == request.user_id,
            Resume.candidate_name == request.candidate_name,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
