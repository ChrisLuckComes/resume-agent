import os
from pathlib import Path

from dotenv import load_dotenv
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


load_dotenv()


def _build_database_url() -> str:
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        return configured_url

    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")

    if db_password and db_host and db_name:
        return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}/{db_name}"

    sqlite_path = Path(__file__).resolve().parent / "resume_agent.db"
    return f"sqlite+aiosqlite:///{sqlite_path.as_posix()}"


DATABASE_URL = _build_database_url()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def _build_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis.from_url(redis_url, decode_responses=True)

    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        return None

    return redis.Redis(
        host=redis_host,
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
    )


redis_client = _build_redis_client()
