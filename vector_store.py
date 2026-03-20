import chromadb
from chromadb.config import Settings


# 初始化客户端
client = chromadb.PersistentClient(path="./chroma_db")

# 创建一个集合
collection = client.get_or_create_collection(name="my_resume")

if __name__ == "__main__":
    print(f"向量存储已初始化，集合 {collection.name} 已创建或获取成功。")
