import os
import sys
import chromadb
import asyncio
from google import genai
from dotenv import load_dotenv


class TraderAgent:
    def __init__(self):
        load_dotenv()
        self.client_ai = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )  # Initialize the Gemini API client
        self.model_name = os.getenv("GEMINI_MODEL_NAME")

        self.client_db = chromadb.PersistentClient(
            path="./chroma_db"
        )  # Initialize ChromaDB client
        self.collection = self.client_db.get_or_create_collection(
            name="my_resume"
        )  # Create or get the collection for trading data

        self.system_instruction = """
        你是一个极其专业的职业经纪人
        请基于提供的简历片段回答问题。你要说真话，不要为了讨好用户而过度美化
        如果简历里没写，就直接说不知道，不要瞎编。
        """

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

    async def ask(self, user_query: str):
        # 完整的RAG问答流程
        query_vector = self._get_embedding(user_query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=1)

        # 获取匹配内容
        context = (
            results["documents"][0][0]
            if results["documents"][0]
            else "(未找到相关简历描述)"
        )

        # 构造增强prompt
        rag_prompt = f"""
        【参考简历片段】
        {context}
        【用户问题】
        {user_query}
        """

        # 调用AI生成回答
        response = await self.client_ai.aio.models.generate_content(
            model=self.model_name,
            config={"system_instruction": self.system_instruction},  # 注入instuction
            contents=rag_prompt,
        )
        return response.text


# 程序入口
async def main():
    # 初始化agent
    agent = TraderAgent()

    # 准备简历数据
    resumes = [
        {
            "id": "exp_1",
            "text": "11年研发经验，熟悉架构设计，主导过多个大型项目的开发。",
        },
        {"id": "exp_2", "text": "擅长金融量化交易系统开发，熟悉A股交易机制"},
    ]

    # 同步数据
    agent.sync_data(resumes)

    # 获取用户输入提问
    if len(sys.argv) < 2:
        print("请输入问题，例如：python main.py '帮我找一个了解证券市场业务的开发者'")
        return
    
    user_query = " ".join(sys.argv[1:])
    print(f"正在根据简历数据分析：{user_query}")

    answer = await agent.ask(user_query)
    print(f"AI的回答是：{answer}")

if __name__ == "__main__":
    asyncio.run(main())