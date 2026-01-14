from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
import datetime

Base = declarative_base()

class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # 冷门仓库（文档不足）跳过处理

class ProcessedRepo(Base):
    __tablename__ = "processed_repos"

    id = Column(Integer, primary_key=True, index=True)
    repo_id = Column(String(50), unique=True, index=True)
    repo_name = Column(String(255))
    repo_url = Column(String(500))
    description = Column(Text, nullable=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    error_message = Column(Text, nullable=True)
    feishu_doc_token = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

engine = create_engine("sqlite:///mcp_deepwiki.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
