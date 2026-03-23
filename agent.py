import os
import sys
import chromadb
import asyncio
import json
from google import genai
from dotenv import load_dotenv


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

    # 为每个用户生成独立的文件路径
    def _get_history_path(self, user_id):
        return f"chat_history_{user_id}.json"

    def _load_user_history(self, user_id):
        """从文件加载聊天历史"""
        path = self._get_history_path(user_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_user_history(self, user_id, history):
        path = self._get_history_path(user_id)
        # 只保留最近20条对话，防止历史过长
        if len(history) > 20:
            history = history[-20:]

        """将最新的history对象序列化并保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

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
            user_history = self._load_user_history(user_id)
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
        response = await self.sessions[user_id].send_message(full_input)

        # 5. 持久化：每次对话完，更新磁盘上的记忆
        self.histories[user_id].append(
            {"role": "user", "parts": [{"text": user_query}]}
        )
        self.histories[user_id].append(
            {"role": "model", "parts": [{"text": response.text}]}
        )

        self._save_user_history(user_id, self.histories[user_id])

        return response.text


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
