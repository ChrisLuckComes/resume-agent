import os
import sys
import chromadb
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="my_resume")


def get_embedding(text: str):
    result = client_ai.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values


# 准备简历数据
resumes = [
    {
        "id": "exp_1",
        "text": "11年研发经验，熟悉架构设计，主导过多个大型项目的开发。",
    },
    {"id": "exp_2", "text": "擅长金融量化交易系统开发，熟悉A股交易机制"},
]


async def ask_with_rag(user_query: str):
    # 检索阶段
    query_vector = get_embedding(user_query)
    results = collection.query(query_embeddings=[query_vector], n_results=1)
    context = results["documents"][0][0]  # 获取到最相关的内容

    # 增强阶段
    # 构建一个特殊的prompt，告诉AI只能根据提供的材料回答
    rag_prompt = f"""
    你是一个面试助手，请根据以下提供的【简历片段】来回答用户的问题。
    如果简历中没有相关信息，请诚实回答不知道。
    【简历片段】：
    {context}
    【用户问题】：
    {user_query}
    """

    # 生成阶段
    response = client_ai.models.generate_content(
        model="gemini-3-flash-preview", contents=rag_prompt
    )

    return response.text

async def main():
    if len(sys.argv) < 2:
        print("请输入问题，例如：python main.py '帮我找一个了解证券市场业务的开发者'")
        return
    
    user_query = " ".join(sys.argv[1:])
    response = await ask_with_rag(user_query)
    print(f"AI的回答是：{response}")

if __name__ == "__main__":
    # 存入数据库
    for item in resumes:
        vector = get_embedding(item["text"])
        collection.upsert(ids=[item["id"]], embeddings=[vector], documents=[item["text"]])

    print("简历数据已添加到向量存储中。")

    # query_text = "帮我找一个了解证券市场业务的开发者"
    # query_vector = get_embedding(query_text)

    # results = collection.query(query_embeddings=[query_vector], n_results=1)

    # print(f"最匹配的内容是：{results['documents'][0]}")
    asyncio.run(main())
