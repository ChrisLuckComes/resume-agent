import json
import importlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, TypeVar, cast

from dotenv import load_dotenv
import cv2  # type: ignore[import-not-found]
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from models import (
    ChatMessage,
    ChatSuggestionsResponse,
    EvaluationResult,
    JDAnalysisResponse,
    JDKeywordExtractionResult,
    OCRResponse,
    RadarMetric,
)
from resume_parser import ResumeParser


ModelT = TypeVar("ModelT", bound=BaseModel)


class ResumeAgent:
    def __init__(self, redis_client=None):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

        base_dir = Path(__file__).resolve().parent
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            temperature=0.2,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/text-embedding-004"),
        )
        rapidocr_module = self._load_rapidocr_module()
        self.ocr_engine = rapidocr_module.RapidOCR() if rapidocr_module is not None else None
        self.vector_store = Chroma(
            collection_name="resume_agent",
            embedding_function=self.embeddings,
            persist_directory=str(base_dir / "chroma_db"),
        )
        self.redis_client = redis_client
        self.cache_expire = 3600
        self.parser = ResumeParser(chunk_size=500, overlap=80)
        self.system_instruction = (
            "你是一名专业、冷静、重证据的招聘顾问。"
            "你的判断必须基于提供的简历内容、JD信息和检索片段。"
            "禁止补充候选人未明确写出的经历、技能、业绩或职责，禁止基于常识脑补。"
            "如果证据不足，必须明确说明“证据不足”或“简历中没有直接体现”。"
            "结论要服务招聘决策，语言简洁、直接、可执行，避免空泛表扬。"
            "输出时优先引用具体事实，例如项目、技术、职责、年限、业务场景。"
        )

    async def extract_text_from_image(self, file_bytes: bytes, mime_type: str) -> OCRResponse:
        if self.ocr_engine is None:
            raise ValueError("未安装 rapidocr_onnxruntime，无法执行图片OCR")

        image_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise ValueError("图片解码失败，无法执行OCR")

        result, _ = self.ocr_engine(decoded_image)
        if not result:
            return OCRResponse(text="")

        text = "\n".join(item[1] for item in result if len(item) > 1 and item[1].strip())
        return OCRResponse(text=text.strip())

    async def generate_follow_up_suggestions(
        self,
        question: str,
        answer: str,
        candidate_name: str | None = None,
    ) -> list[str]:
        prompt = ChatPromptTemplate.from_template(
            """
            你是招聘顾问，请根据本轮问答生成 3 条适合继续追问的问题。
            候选人：{candidate_name}
            用户问题：{question}
            AI回答：{answer}

            要求：
            1. 每条都是中文追问句子
            2. 3条问题分别优先覆盖：技术深度、真实职责或ownership、风险确认
            3. 要紧扣当前问答里已经出现的信息，不要泛泛提问
            4. 避免重复，避免空话，避免一次问太多点
            """
        )
        try:
            structured_llm = self.llm.with_structured_output(ChatSuggestionsResponse)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "candidate_name": candidate_name or "未指定",
                    "question": question,
                    "answer": answer,
                }
            )
            result: ChatSuggestionsResponse = self._coerce_model(
                raw_result, ChatSuggestionsResponse
            )
            suggestions = self._unique_strings(result.suggestions)
            if suggestions:
                return suggestions[:3]
        except Exception:
            pass

        base_name = candidate_name or "候选人"
        return [
            f"{base_name} 在最近一段项目中的具体职责是什么？",
            f"{base_name} 在关键技术方案里承担了多大 ownership？",
            f"针对当前岗位要求，{base_name} 最大的风险点是什么？",
        ]

    async def analyze_jd(self, jd_text: str) -> JDAnalysisResponse:
        cleaned_text = jd_text.strip()
        if not cleaned_text:
            return JDAnalysisResponse(keywords=[])

        return JDAnalysisResponse(keywords=await self._extract_jd_keywords(cleaned_text))

    async def _extract_jd_keywords(self, text: str) -> list[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是招聘JD关键词抽取助手。你的唯一任务是从输入原文中提取关键词。"
                    "禁止补充原文未出现的词，禁止改写为同义词，禁止根据常识推断。"
                    "只保留文本中能直接定位到的技能、工具、业务领域、职责方向、方法论等关键词。"
                    "优先保留对招聘筛选最有区分度的词，过滤'负责'、'沟通能力'这类泛词。"
                    "如果原文中出现英文、缩写、大小写混合写法，尽量保留原文写法。"
                    "不要输出完整句子，不要输出解释。"
                    "关键词数量控制在4到8个，若有效关键词不足则按实际数量返回。"
                ),
                (
                    "user",
                    "请从下面JD原文中提取关键词，仅返回结构化结果。\n\nJD原文：\n{text}",
                ),
            ]
        )

        try:
            structured_llm = self.llm.with_structured_output(JDKeywordExtractionResult)
            raw_result = await (prompt | structured_llm).ainvoke({"text": text})
            result: JDKeywordExtractionResult = self._coerce_model(
                raw_result, JDKeywordExtractionResult
            )
            keywords = self._normalize_keyword_candidates(result.keywords, text)
            if keywords:
                return keywords[:8]
        except Exception:
            pass

        return self._fallback_keywords(text)

    async def evaluate_resume(
        self,
        resume_text: str,
        jd_text: str,
        jd_keywords: list[str] | None = None,
    ) -> dict:
        extracted_keywords = await self._extract_jd_keywords(jd_text)
        keywords = self._unique_strings(jd_keywords or extracted_keywords)

        try:
            structured_llm = self.llm.with_structured_output(EvaluationResult)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.system_instruction
                        + " 输出结构必须适合招聘评审，评分范围为0到100。"
                        + " 亮点和风险都必须基于简历中的明确证据，禁止写空泛结论。",
                    ),
                    (
                        "user",
                        """
                        请结合以下信息评估候选人：
                        JD关键词: {keywords}

                        JD全文:
                        {jd_text}

                        简历全文:
                        {resume_text}

                        评估原则：
                        1. 只能依据简历中明确出现的信息判断。
                        2. 如果JD某项要求在简历中没有直接证据，必须视为证据不足，不能脑补。
                        3. 亮点必须具体到技能、项目、职责、结果或业务场景。
                        4. 风险必须具体指出缺口、不确定项或需要面试确认的点。
                        5. 不要使用“潜力不错”“综合素质较强”这类空泛表述。

                        评分参考：
                        - 90到100：核心技能、项目复杂度、业务场景均高度匹配，且证据充分
                        - 75到89：大部分要求匹配，存在少量缺口但整体可推进
                        - 60到74：有相关经历，但关键要求证据不足或存在明显短板
                        - 0到59：核心要求不匹配，或缺少支持结论的直接证据

                        请返回：
                        - summary: 2到3句中文结论
                        - title: 一句简短标题
                        - decision: 明确结论
                        - match_score: 0到100整数
                        - radar_metrics: 4到6个维度，每个维度包含name、value、max
                        - highlights: 3到5条亮点
                        - risks: 2到4条风险
                        """,
                    ),
                ]
            )
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                }
            )
            result: EvaluationResult = self._coerce_model(raw_result, EvaluationResult)
            payload = result.model_dump()
            if not payload.get("radar_metrics"):
                payload["radar_metrics"] = [
                    metric.model_dump()
                    for metric in self._build_radar_metrics(payload["match_score"])
                ]
            return payload
        except Exception:
            return self._fallback_evaluation(resume_text, jd_text, keywords)

    async def ingest_resume(
        self,
        file_name: str,
        file_content: bytes,
        user_id: str,
        resume_id: int,
        candidate_name: str,
        phone: str,
    ) -> str:
        suffix = Path(file_name).suffix.lower() or ".docx"
        if suffix not in {".docx", ".pdf"}:
            raise ValueError("仅支持 .docx 或 .pdf 格式的简历文件")

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            raw_text = self.parser.extract_text(temp_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if not raw_text.strip():
            raise ValueError("简历内容为空，无法继续处理")

        chunks = self.parser.get_chunks(raw_text)
        self.delete_resume_vectors(user_id=user_id, resume_id=resume_id)
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=[
                {
                    "user_id": user_id,
                    "resume_id": str(resume_id),
                    "candidate_name": candidate_name,
                    "phone": phone,
                }
                for _ in chunks
            ],
            ids=[f"{user_id}:{resume_id}:{index}" for index in range(len(chunks))],
        )
        return raw_text

    def delete_resume_vectors(
        self,
        user_id: str,
        resume_id: int | None = None,
        candidate_name: str | None = None,
    ) -> None:
        where = self._build_vector_filter(
            user_id=user_id, resume_id=resume_id, candidate_name=candidate_name
        )
        if where:
            self.vector_store.delete(where=where)

    async def ask(
        self,
        question: str,
        user_id: str,
        db: AsyncSession,
        candidate_name: str | None = None,
        resume_id: int | None = None,
    ) -> str:
        chunks: list[str] = []
        async for chunk in self.stream_chat(
            question=question,
            user_id=user_id,
            db=db,
            candidate_name=candidate_name,
            resume_id=resume_id,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream_chat(
        self,
        question: str,
        user_id: str,
        db: AsyncSession,
        candidate_name: str | None = None,
        resume_id: int | None = None,
    ) -> AsyncIterator[str]:
        chat_history = await self._load_history(user_id=user_id, db=db)
        context = await self._retrieve_context(
            question=question,
            user_id=user_id,
            candidate_name=candidate_name,
            resume_id=resume_id,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_instruction
                    + " 如果检索内容不足以支撑结论，就明确说明证据不足。"
                    + " 优先回答结论，再补充证据；如果用户问是否匹配、是否具备某能力，"
                    + "要先给判断，再给依据，最后指出需要进一步确认的风险。",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "相关简历片段:\n{context}\n\n"
                    "回答要求:\n"
                    "1. 只基于相关简历片段回答。\n"
                    "2. 如果问题是判断类问题，优先按“结论 -> 证据 -> 风险/待确认点”作答。\n"
                    "3. 如果问题是信息查询类问题，直接列出找到的事实。\n"
                    "4. 不要假装看到了上下文之外的信息。\n\n"
                    "用户问题:\n{question}",
                ),
            ]
        )
        chain = prompt | self.llm

        response_parts: list[str] = []
        try:
            async for chunk in chain.astream(
                {
                    "chat_history": chat_history,
                    "context": context or "未检索到匹配的简历片段。",
                    "question": question,
                }
            ):
                text = self._chunk_to_text(chunk)
                if text:
                    response_parts.append(text)
                    yield text
        except Exception:
            fallback = self._build_chat_fallback(question=question, context=context)
            response_parts.append(fallback)
            yield fallback

        full_response = "".join(response_parts).strip()
        await self._save_message(user_id=user_id, role="user", content=question, db=db)
        await self._save_message(user_id=user_id, role="ai", content=full_response, db=db)

    async def _retrieve_context(
        self,
        question: str,
        user_id: str,
        candidate_name: str | None,
        resume_id: int | None,
    ) -> str:
        search_kwargs: dict[str, object] = {"k": 6}
        where = self._build_vector_filter(
            user_id=user_id, resume_id=resume_id, candidate_name=candidate_name
        )
        if where:
            search_kwargs["filter"] = where

        try:
            docs = await self.vector_store.as_retriever(search_kwargs=search_kwargs).ainvoke(
                question
            )
        except Exception:
            return ""

        return "\n\n".join(doc.page_content for doc in docs)

    async def _save_message(
        self, user_id: str, role: str, content: str, db: AsyncSession
    ) -> None:
        db.add(ChatMessage(user_id=user_id, role=role, content=content))
        await db.flush()
        await self._cache_delete(await self._get_cache_key(user_id))

    async def _load_history(self, user_id: str, db: AsyncSession) -> list[BaseMessage]:
        cache_key = await self._get_cache_key(user_id)
        cached = await self._cache_get(cache_key)
        if cached:
            return self._history_from_cache(cached)

        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(12)
        )
        result = await db.execute(stmt)
        messages = cast(list[ChatMessage], result.scalars().all())
        await self._cache_set(
            cache_key,
            json.dumps(
                [{"role": message.role, "content": message.content} for message in messages],
                ensure_ascii=False,
            ),
        )
        return [self._to_langchain_message(message.role, message.content) for message in messages]

    async def _get_cache_key(self, user_id: str) -> str:
        return f"chat_cache:{user_id}"

    async def _cache_get(self, key: str) -> str | None:
        if not self.redis_client:
            return None
        return await self.redis_client.get(key)

    async def _cache_set(self, key: str, value: str) -> None:
        if self.redis_client:
            await self.redis_client.setex(key, self.cache_expire, value)

    async def _cache_delete(self, key: str) -> None:
        if self.redis_client:
            await self.redis_client.delete(key)

    def _history_from_cache(self, cached: str) -> list[BaseMessage]:
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            return []

        history: list[BaseMessage] = []
        for item in payload:
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if content:
                history.append(self._to_langchain_message(role, content))
        return history

    def _coerce_model(self, value: Any, model_type: type[ModelT]) -> ModelT:
        if isinstance(value, model_type):
            return value
        if isinstance(value, dict):
            return model_type.model_validate(value)
        if hasattr(value, "model_dump"):
            return model_type.model_validate(value.model_dump())
        return model_type.model_validate(value)

    def _load_rapidocr_module(self):
        try:
            return importlib.import_module("rapidocr_onnxruntime")
        except ImportError:
            return None

    def _to_langchain_message(self, role: str, content: str) -> BaseMessage:
        if role == "user":
            return HumanMessage(content=content)
        return AIMessage(content=content)

    def _build_vector_filter(
        self,
        user_id: str,
        resume_id: int | None = None,
        candidate_name: str | None = None,
    ) -> dict | None:
        clauses = [{"user_id": user_id}]
        if resume_id is not None:
            clauses.append({"resume_id": str(resume_id)})
        if candidate_name:
            clauses.append({"candidate_name": candidate_name})

        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _fallback_keywords(self, text: str) -> list[str]:
        stopwords = {
            "负责",
            "熟悉",
            "能力",
            "经验",
            "优先",
            "相关",
            "以上",
            "具备",
            "参与",
            "本科",
            "大专",
            "岗位",
            "职位",
            "工作",
            "公司",
            "团队",
            "产品",
            "业务",
            "要求",
            "能够",
            "我们",
            "你将",
        }
        library = [
            "Python",
            "Java",
            "Go",
            "C++",
            "JavaScript",
            "TypeScript",
            "React",
            "Vue",
            "Node.js",
            "SQL",
            "MySQL",
            "PostgreSQL",
            "Redis",
            "Kafka",
            "Docker",
            "Kubernetes",
            "LangChain",
            "RAG",
            "LLM",
            "AI",
            "机器学习",
            "深度学习",
            "数据分析",
            "项目管理",
            "沟通协作",
        ]
        matched_library = [item for item in library if item.lower() in text.lower()]
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}|[\u4e00-\u9fff]{2,8}", text)
        keywords = [token for token in tokens if token not in stopwords and len(token) >= 2]
        return self._unique_strings([*matched_library, *keywords])[:8]

    def _normalize_keyword_candidates(self, candidates: Iterable[str], text: str) -> list[str]:
        normalized: list[str] = []
        for candidate in candidates:
            keyword = candidate.strip().strip("，,。；;:：()[]{}")
            if len(keyword) < 2 or len(keyword) > 30:
                continue

            match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)
            if match:
                normalized.append(match.group(0).strip())

        return self._unique_strings(normalized)

    def _fallback_evaluation(
        self, resume_text: str, jd_text: str, jd_keywords: Iterable[str]
    ) -> dict:
        lowered_resume = resume_text.lower()
        keywords = self._unique_strings(list(jd_keywords))
        matched = [keyword for keyword in keywords if keyword.lower() in lowered_resume]
        coverage = len(matched) / max(len(keywords), 1)
        match_score = max(45, min(95, round(coverage * 100))) if keywords else 60
        title = "高度匹配" if match_score >= 80 else "有一定匹配度" if match_score >= 65 else "建议谨慎评估"
        summary = (
            f"候选人与JD的整体匹配度约为 {match_score} 分。"
            f"当前证据主要来自简历中出现的关键词：{', '.join(matched[:4]) or '暂未识别到明确重合项'}。"
        )
        highlights = [
            f"简历中明确出现关键词：{keyword}" for keyword in matched[:4]
        ] or ["简历已成功解析，可继续结合项目经历人工复核。"]
        risks = []
        missing = [keyword for keyword in keywords if keyword not in matched]
        if missing:
            risks.append(f"以下JD关键词在简历中缺少直接证据：{', '.join(missing[:4])}")
        risks.append("当前结果为降级评估，建议结合面试继续确认细节。")
        radar_metrics = [
            metric.model_dump()
            for metric in self._build_radar_metrics(match_score)
        ]
        return {
            "summary": summary,
            "title": title,
            "decision": title,
            "match_score": match_score,
            "radar_metrics": radar_metrics,
            "highlights": highlights,
            "risks": risks,
        }

    def _build_radar_metrics(self, match_score: int) -> list[RadarMetric]:
        metric_names = ["技术深度", "项目经验", "软技能", "背景示例", "AI技能"]
        offsets = [0, -6, -10, -4, -8]
        metrics: list[RadarMetric] = []
        for index, name in enumerate(metric_names):
            value = max(0, min(100, match_score + (offsets[index] if index < len(offsets) else -8)))
            metrics.append(RadarMetric(name=name, value=value))
        return metrics

    def _chunk_to_text(self, chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _build_chat_fallback(self, question: str, context: str) -> str:
        if context:
            snippet = context[:500]
            return (
                "模型流式调用失败，以下是基于已检索简历片段的降级回答："
                f"\n问题：{question}\n简历证据：{snippet}"
            )
        return "暂时无法生成回答，因为没有检索到可用的简历上下文。"

    def _unique_strings(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result
