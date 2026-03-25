import os
from dotenv import load_dotenv
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base


# 加载环境变量
load_dotenv()

# 从环境变量或配置中获取
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
# ... 其他配置 ...

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# 1. 创建全局唯一的异步引擎
engine = create_async_engine(DATABASE_URL, echo=False)

# 2. 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

# 3. 基础类，供 models.py 继承
Base = declarative_base()


# 4. 关键的依赖注入函数
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # 如果程序能执行到这里，说明业务逻辑没有抛出异常
            await session.commit()
        except Exception:
            # 如果中间任何环节报错，自动回滚以保证数据一致性
            await session.rollback()
            raise
        finally:
            # 无论成功还是失败，最后都要关闭连接
            await session.close()


redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=6379,
    db=0,
    decode_responses=True,  # 自动把bytes转换成str
)