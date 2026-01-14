import os
import asyncio
import logging
import uvicorn
import datetime
from fastapi import FastAPI, BackgroundTasks, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import glob

from models import SessionLocal, init_db, ProcessedRepo, ProcessingStatus
from github_client import GitHubMonitor
from feishu_client import FeishuService
from mcp_client import DeepWikiMCPClient
from rag_refine import RAGRefiner, Config
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_deepwiki.log"),
        logging.StreamHandler()
    ],
    force=True  # override any previous logging.basicConfig (e.g., from imports)
)
logger = logging.getLogger(__name__)

# Task running flag to prevent concurrent sync tasks
is_task_running = False

# Initialize DB
init_db()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Service Instances
github_monitor = None
feishu_service = None
rag_refiner = None
mcp_client = None
deepwiki_indexer = None
templates = Jinja2Templates(directory="/www/wwwroot/mcp_deepwiki/templates")

config = Config()

async def process_repo_workflow(db: Session, repo_data: dict):
    repo_id = str(repo_data["id"])
    repo_name = repo_data["full_name"]
    github_url = repo_data["html_url"]
    
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    
    if db_repo:
        # Always update timestamp to show we checked it
        db_repo.updated_at = datetime.datetime.now(datetime.UTC)
        db.commit()
        
    # Skip COMPLETED, PROCESSING, and SKIPPED repos, but allow retrying FAILED and PENDING repos
    if db_repo and (db_repo.status == ProcessingStatus.COMPLETED or db_repo.status == ProcessingStatus.PROCESSING or db_repo.status == ProcessingStatus.SKIPPED):
        return
        
    if not db_repo:
        db_repo = ProcessedRepo(
            repo_id=repo_id,
            repo_name=repo_name,
            repo_url=github_url,
            description=repo_data.get("description"),
            status=ProcessingStatus.PROCESSING
        )
        db.add(db_repo)
    else:
        db_repo.status = ProcessingStatus.PROCESSING
    db.commit()
    db.refresh(db_repo)
    
    try:
        logger.info(f"🚀 开始处理仓库: {repo_name}")
        
        safe_name = repo_name.replace("/", "_")
        base_dir = f"/www/wwwroot/mcp_deepwiki/output/{safe_name}"
        
        # 1. Fetch from DeepWiki MCP if data is missing
        if not os.path.exists(base_dir) or not glob.glob(os.path.join(base_dir, "*Overview.md")):
            logger.info(f"📥 数据缺失，从 DeepWiki MCP 获取: {repo_name}")
            try:
                await mcp_client.fetch_and_save(repo_name)
            except Exception as e:
                # If MCP fails, it's likely unindexed or a connection issue. Skip for now to avoid dead loops.
                raise Exception(f"MCP fetch failed: {e}. The repository might not be indexed in DeepWiki.")

        if not os.path.exists(base_dir):
            raise Exception(f"Repo data folder not found after MCP fetch: {base_dir}")

        # Find Overview file
        overview_files = glob.glob(os.path.join(base_dir, "*Overview.md"))

        # Check if it's a cold repository (only has 1 document file)
        all_md_files = glob.glob(os.path.join(base_dir, "*.md"))
        if len(all_md_files) <= 1:
            logger.warning(f"⚠️ 冷门仓库检测：{repo_name} 只有 {len(all_md_files)} 个文档，标记为跳过")
            db_repo.status = ProcessingStatus.SKIPPED
            db_repo.error_message = f"冷门仓库：仅有 {len(all_md_files)} 个文档（需要 Overview.md）"
            db.commit()
            return

        if not overview_files:
            raise Exception("No Overview.md found")
        target_file = overview_files[0]

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. RAG Refine
        logger.info(f"✍️ 正在生成初稿...")
        # Generate Draft
        draft = await rag_refiner.generate_draft(content)
        
        # Select documents
        logger.info(f"📚 正在选择相关文档...")
        candidate_files = [os.path.basename(p) for p in glob.glob(os.path.join(base_dir, "*.md")) if os.path.abspath(p) != os.path.abspath(target_file)]
        selected_files = await rag_refiner.select_documents_for_rag(draft, candidate_files)

        # Build Knowledge Base
        logger.info(f"🧠 正在构建向量知识库...")
        await rag_refiner.build_knowledge_base(base_dir, target_file, selected_files)

        # Final Expand
        logger.info(f"🔄 正在通过 RAG 扩展内容...")
        final_content = await rag_refiner.expand_with_rag(draft)
        
        # 3. Upload to Feishu
        logger.info(f"📤 正在上传到飞书知识库...")
        title = f"{repo_name} RAG Refined"
        if not db_repo.feishu_doc_token:
            logger.info(f"🆕 创建新的飞书文档节点")
            doc_token = await feishu_service.create_node(title=title)
            if doc_token:
                db_repo.feishu_doc_token = doc_token
                db.commit()
        else:
            logger.info(f"📝 更新已有飞书文档")
            doc_token = db_repo.feishu_doc_token

        if doc_token:
            # Note: update_document_content currently appends content.
            # In a production scenario, you might want to clear existing blocks first.
            await feishu_service.update_document_content(doc_token, final_content)

            # 4. Notify
            logger.info(f"🔔 发送通知...")
            await feishu_service.send_card_notification(
                title=f"RAG Refined Wiki: {repo_name}",
                summary=repo_data.get("description") or "Documentation optimized via RAG workflow.",
                url=f"https://feishu.cn/docx/{doc_token}"
            )
            # Add plain text webhook notification
            await feishu_service.send_webhook_notification(repo_name, doc_token)
        
        logger.info(f"✅ 仓库处理完成: {repo_name}")
        db_repo.status = ProcessingStatus.COMPLETED
    except Exception as e:
        error_msg = str(e)
        # Check if it's a cold repository error
        is_cold_repo = (
            "No Overview.md found" in error_msg or
            ("MCP fetch failed" in error_msg and "unindexed" in error_msg.lower()) or
            ("TaskGroup" in error_msg and "sub-exception" in error_msg)
        )

        if is_cold_repo:
            logger.warning(f"⚠️ 冷门仓库 [{repo_name}]: {error_msg}")
            db_repo.status = ProcessingStatus.SKIPPED
            db_repo.error_message = f"冷门仓库：{error_msg}"
        else:
            logger.error(f"❌ 处理失败 [{repo_name}]: {error_msg}")
            db_repo.status = ProcessingStatus.FAILED
            db_repo.error_message = error_msg
    
    db.commit()

async def sync_task(sync_all: bool = False, silent: bool = False):
    global is_task_running

    # Check if another task is already running
    if is_task_running:
        if not silent:
            logger.info(f"⏳ 另一个任务正在运行，跳过此次调度")
        return

    is_task_running = True

    try:
        if not silent:
            logger.info(f"🔄 开始同步任务 (sync_all={sync_all})")
        db = SessionLocal()
        try:
            # 1. Fetch new star repositories from GitHub
            logger.info(f"⭐ 正在获取 GitHub 最新 star...")
            stars = await github_monitor.fetch_recent_stars(limit=10)
            logger.info(f"📦 发现 {len(stars)} 个新的 star 仓库")
            for star in stars:
                await process_repo_workflow(db, star)

            # 2. Process pending/failed repositories from database (only FAILED and PENDING, not SKIPPED)
            pending_repos = db.query(ProcessedRepo).filter(
                (ProcessedRepo.status == ProcessingStatus.PENDING) |
                (ProcessedRepo.status == ProcessingStatus.FAILED)
            ).all()

            if pending_repos:
                if not silent:
                    logger.info(f"📋 发现 {len(pending_repos)} 个待处理/失败的历史仓库")
            elif not silent:
                logger.info(f"✨ 没有待处理的历史仓库")

            for repo in pending_repos:
                # Convert db record to dict format expected by process_repo_workflow
                repo_data = {
                    "id": repo.repo_id,
                    "full_name": repo.repo_name,
                    "html_url": repo.repo_url,
                    "description": repo.description
                }
                await process_repo_workflow(db, repo_data)
        finally:
            db.close()
            if not silent:
                logger.info("✅ 同步任务完成")
    finally:
        is_task_running = False

# Background Scheduler
async def scheduler_loop():
    while True:
        await asyncio.sleep(30) # Run every 30 seconds
        await sync_task(sync_all=False, silent=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global github_monitor, feishu_service, rag_refiner, mcp_client
    logger.info("=" * 50)
    logger.info("🚀 MCP DeepWiki 服务启动中...")
    logger.info("=" * 50)

    github_monitor = GitHubMonitor(os.getenv("GITHUB_TOKEN"))
    logger.info("✅ GitHub 监控器已初始化")

    feishu_service = FeishuService(
        os.getenv("FEISHU_APP_ID"),
        os.getenv("FEISHU_APP_SECRET"),
        os.getenv("FEISHU_SPACE_ID"),
        os.getenv("FEISHU_WEBHOOK_URL")
    )
    logger.info("✅ 飞书服务已初始化")

    rag_refiner = RAGRefiner()
    logger.info("✅ RAG 精炼器已初始化")

    mcp_client = DeepWikiMCPClient()
    logger.info("✅ DeepWiki MCP 客户端已初始化")

    logger.info("⏰ 启动后台调度器 (每30秒执行一次)")
    asyncio.create_task(scheduler_loop())

    # Initialize DB with historical stars on first run
    db = SessionLocal()
    try:
        if db.query(ProcessedRepo).count() == 0:
            logger.info("🎯 首次运行：正在初始化数据库，导入所有历史 star 仓库...")
            stars = await github_monitor.fetch_all_stars()
            logger.info(f"📊 共找到 {len(stars)} 个 star 仓库")
            for star in stars:
                repo_id = str(star["id"])
                if not db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first():
                    # Check if we already have the output folder for this repo
                    safe_name = star["full_name"].replace("/", "_")
                    base_dir = f"/www/wwwroot/mcp_deepwiki/output/{safe_name}"

                    status = ProcessingStatus.PENDING
                    # If refined file already exists, mark as completed
                    if os.path.exists(os.path.join(base_dir, "refined", "02_Overview_Refined.md")):
                        status = ProcessingStatus.COMPLETED

                    repo = ProcessedRepo(
                        repo_id=repo_id,
                        repo_name=star["full_name"],
                        repo_url=star["html_url"],
                        description=star.get("description"),
                        status=status
                    )
                    db.add(repo)
            db.commit()
            logger.info(f"✅ 数据库初始化完成，共 {len(stars)} 个仓库")
        else:
            logger.info("✅ 数据库已初始化，跳过首次运行设置")
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
    finally:
        db.close()

    logger.info("=" * 50)
    logger.info("🎉 MCP DeepWiki 服务启动完成！")
    logger.info("=" * 50)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/repos")
async def get_repos(db: Session = Depends(get_db)):
    return db.query(ProcessedRepo).order_by(ProcessedRepo.updated_at.desc()).all()

@app.post("/api/retry/{repo_id}")
async def retry_repo(repo_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    if not db_repo:
        logger.warning(f"⚠️ 重试失败：仓库 {repo_id} 不存在")
        return {"error": "Repository not found"}, 404

    logger.info(f"🔄 正在重试仓库: {db_repo.repo_name} (当前状态: {db_repo.status.value})")
    # Reset status to PENDING to allow it to be picked up (works for FAILED and SKIPPED)
    db_repo.status = ProcessingStatus.PENDING
    db_repo.error_message = None
    db_repo.updated_at = datetime.datetime.now(datetime.UTC)
    db.commit()

    # Trigger a sync task in background to process immediately
    background_tasks.add_task(sync_task, False, True)

    logger.info(f"✅ 已将仓库 {db_repo.repo_name} 标记为待处理")
    return {"status": "retrying"}

@app.post("/trigger")
async def trigger(background_tasks: BackgroundTasks, sync_all: bool = False):
    logger.info(f"🎯 手动触发同步任务 (sync_all={sync_all})")
    background_tasks.add_task(sync_task, sync_all)
    return {"status": "triggered"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
