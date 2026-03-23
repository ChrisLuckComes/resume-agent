import os
import sys
import chromadb
import asyncio
import json
import redis.asyncio as redis
from google import genai
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, Column, Integer, String, Text, DateTime, select
from datetime import datetime

# 定义Base ORM基类
Base = declarative_base()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), index=True)
    role = Column(String(50))  # 'user' 或 'model'
    content = Column(Text)  # 消息内容
    created_at = Column(DateTime, default=datetime.utcnow)  # 消息时间戳


def get_project_tech_stack(project_name: str):
    """
    当用户询问某个项目的具体技术细节（如使用的中间件、数据库、框架）时，调用此函数获取详细信息。

    Args:
        project_name: 项目关键词，例如'架构设计'或'金融量化'
    """

    # 模拟一个从更深的数据库中检索更详细信息的过程
    detailed_db = {
        "架构设计": "使用微服务架构，主要技术栈包括Spring Boot, Docker, Kubernetes。",
        "金融量化": "使用Python进行量化策略开发，主要技术栈包括Pandas, NumPy, scikit-learn。",
    }

    return detailed_db.get(project_name, "该项目没有更详细的技术文档记录。")


def get_current_date(year: str):
    """
    当用户询问当前日期或时间相关的问题时，调用此函数获取当前日期。

    Args:
        year: 用户输入的年份关键词，例如'11年研发经验'中的'11年'，实际上当前日期可能不止11年了，所以需要获取当前年份来计算实际经验年限，简历中写了毕业年份2014年，所以可以通过当前年份减去2014年来计算实际经验年限。
    """
    from datetime import datetime

    return datetime.now().year - 2014  # 2014是简历中毕业的年份


class ResumeAgent:
    def __init__(self, history_file="chat_history.json"):
        load_dotenv()
        self.client_ai = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )  # Initialize the Gemini API client

        self.model_name = os.getenv("GEMINI_MODEL_NAME")

        # 设定静态存储路径，使用ChromaDB来存储简历数据和对应的向量
        self.client_db = chromadb.PersistentClient(
            path="./chroma_db"
        )  # Initialize ChromaDB client
        self.collection = self.client_db.get_or_create_collection(
            name="my_resume"
        )  # Create or get the collection for trading data

        # 设定system instruction，明确AI的角色和行为准则
        self.system_instruction = """
        你是一个极其专业的职业经纪人
        请基于提供的简历片段回答问题。你要说真话，不要为了讨好用户而过度美化
        如果简历里没写，就直接说不知道，不要瞎编。
        """

        # 定义一个变量存储会话，初始为 None
        self.sessions = {}  # 结构 {"user:id": chat_session_object}

        # 加载持久化的历史纪录到内存
        # self.history_file = history_file
        # self.history = self._load_history()
        self.histories = {}  # 结构 {"user:id": [messages...]}

        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")

        self.db_url = (
            f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}/{db_name}"
        )

        # 创建异步数据库引擎
        self.engine = create_async_engine(self.db_url, echo=False)

        # 创建会话工厂
        self.AysncSessionLocal = async_sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # 初始化redis连接
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=6379,
            db=0,
            decode_responses=True,  # 自动把bytes转换成str
        )

        self.cache_expire = 3600  # 缓存1小时

    async def _get_cache_key(self, user_id: str):
        return f"chat_cache:{user_id}"

    async def init_db(self):
        """初始化数据库"""
        async with self.engine.begin() as conn:
            # 使用SQLAlchemy的模型映射自动建表
            await conn.run_sync(Base.metadata.create_all)
            print("数据库表结构完成")

    async def _save_message(self, user_id: str, role: str, content: str):
        """双写逻辑：PG 永久存，Redis 更新缓存"""

        # 1. 异步写入PostgreSQL数据库
        async with self.AysncSessionLocal() as session:
            async with session.begin():  # 自动处理begin和commit/rollback
                new_message = ChatMessage(
                    user_id=user_id,
                    role=role,
                    content=content,
                )
                session.add(new_message)
            # 离开async with时，session会自动关闭
        
        # 2. 更新Redis缓存
        cache_key = await self._get_cache_key(user_id)
        await self.redis_client.delete(cache_key)  # 删除旧缓存，下一次查询会从PG加载最新数据

    async def _load_history(self, user_id, limit: int = 10):
        """双级缓存查询 Redis->PG"""
        cache_key = await self._get_cache_key(user_id)

        # 1. 先查Redis缓存
        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            print(f"从Redis缓存中加载用户 {user_id} 的历史记录")
            return json.loads(cached_data)

        """ redis没命中，从PG加载最近的历史记录"""
        print(f"[PG]Redis未命中，从数据库中读取用户 {user_id} 的历史记录")
        async with self.AysncSessionLocal() as session:
            # 查最近的N条，按时间降序排列
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.user_id == user_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()
            # 转换成模型需要的格式，注意要逆序（从旧到新）
            history = [
                {"role": msg.role, "parts": [{"text": msg.content}]}
                for msg in reversed(messages)
            ]

            if history:
                # 加载到Redis缓存，设置过期时间
                await self.redis_client.setex(
                    cache_key, self.cache_expire, json.dumps(history)
                )

            return history

    def _get_embedding(self, text: str):
        result = self.client_ai.models.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME"), contents=text
        )
        return result.embeddings[0].values

    def sync_data(self, data_list):
        # 同步数据到数据库
        for item in data_list:
            vector = self._get_embedding(item["text"])
            self.collection.upsert(
                ids=[item["id"]],
                embeddings=[vector],
                documents=[item["text"]],
            )
        print("数据同步完成")

    async def ask(self, user_query: str, user_id: str):
        # 1. 如果会话没有创建，尝试从磁盘恢复
        if user_id not in self.sessions:
            print(f"为用户 {user_id} 创建新会话")
            user_history = await self._load_history(user_id)
            self.histories[user_id] = user_history

            self.sessions[user_id] = self.client_ai.aio.chats.create(
                model=self.model_name,
                history=user_history,  # 关键：把历史喂给新会话
                config={
                    "system_instruction": self.system_instruction,
                    "tools": [get_project_tech_stack, get_current_date],
                },
            )

        # 2. 完整的RAG问答流程
        query_vector = self._get_embedding(user_query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=1)
        context = (
            results["documents"][0][0]
            if results["documents"][0]
            else "(未找到相关简历描述)"
        )

        # 3. 构造增强prompt
        full_input = f"【参考背景】：{context}\n\n【用户问题】：{user_query}"

        # 4. 使用chat_session.send_message来发送消息
        try:
            response = await self.sessions[user_id].send_message(full_input)
            response_text = response.text
        except Exception as e:
            print(f"Error while sending message: {e}")
            response_text = f"对不起，处理您的请求时出错。可能是API超额，这是模拟恢复，我已收到问题:{user_query}"

        # 5. 持久化：每次对话完，更新磁盘上的记忆
        self.histories[user_id].append(
            {"role": "user", "parts": [{"text": user_query}]}
        )
        self.histories[user_id].append(
            {"role": "model", "parts": [{"text": response_text}]}
        )

        # 6. 异步写入数据库
        await self._save_message(user_id, "user", user_query)
        await self._save_message(user_id, "model", response_text)

        return response_text


# 程序入口
# async def main():
#     # 初始化agent
#     agent = ResumeAgent()

#     print("AI职业经纪人已启动，输入'exit'退出")

#     # 准备简历数据
#     resumes = [
#         {
#             "id": "exp_1",
#             "text": "9年研发经验，熟悉架构设计，主导过多个大型项目的开发。",
#         },
#         {"id": "exp_2", "text": "擅长金融量化交易系统开发，熟悉A股交易机制"},
#     ]

#     # 同步数据
#     agent.sync_data(resumes)

#     while True:
#         user_input = input("请输入问题：")
#         if user_input.lower() == "exit":
#             print("退出程序")
#             break
#         answer = await agent.ask(user_input, user_id)
#         print(f"AI的回答是：{answer}")


# if __name__ == "__main__":
#     asyncio.run(main())
