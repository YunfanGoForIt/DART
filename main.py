import os
import asyncio
import logging
import uvicorn
import datetime
from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import glob
from enum import Enum
from typing import Dict, Any, Optional

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


class TaskType(Enum):
    SYNC = "sync"
    REGENERATE = "regenerate"


class TaskQueue:
    """任务队列管理器 - 单仓库粒度"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_processing = False
        self.current_task = None
        self.task_list = []  # 保存所有任务的列表（用于查询）
        self.repo_ids_in_queue = set()  # 记录队列中的仓库ID，用于去重

    async def add_repo_task(self, task_type: TaskType, repo_data: Dict[str, Any]):
        """添加单个仓库任务到队列"""
        # 兼容两种格式：GitHub API 格式用 "id"，数据库格式用 "repo_id"
        repo_id = repo_data.get("repo_id") or str(repo_data.get("id", ""))

        # 检查是否已在队列中
        if repo_id in self.repo_ids_in_queue:
            return False

        task = {
            "type": task_type,
            "data": repo_data,
            "added_at": datetime.datetime.now()
        }

        await self.queue.put(task)
        self.task_list.append(task)
        self.repo_ids_in_queue.add(repo_id)

        # 定期清理
        self._cleanup_completed_tasks()

        # 获取仓库名称用于日志显示
        repo_name = repo_data.get("full_name") or repo_data.get("repo_name", "Unknown")
        queue_size = self.queue.qsize()
        logger.info(f"📥 仓库已加入队列: {repo_name} (队列长度: {queue_size})")
        return True

    def is_repo_in_queue(self, repo_id: str) -> bool:
        """检查仓库是否已在队列中"""
        return repo_id in self.repo_ids_in_queue

    def _cleanup_completed_tasks(self):
        """清理已完成的任务（保留最近100个）"""
        if len(self.task_list) > 100:
            self.task_list = self.task_list[-100:]

    async def get_next_task(self):
        """获取下一个任务"""
        task = await self.queue.get()
        self.current_task = task
        return task

    def mark_task_done(self):
        """标记当前任务完成"""
        if self.current_task:
            # 兼容两种格式：GitHub API 用 "id"，重新生成用 "repo_id"
            repo_id = self.current_task["data"].get("repo_id") or str(self.current_task["data"].get("id", ""))
            if repo_id:
                self.repo_ids_in_queue.discard(repo_id)
                logger.info(f"✅ 任务完成，已从队列移除: {repo_id}")

            # 从列表中移除已完成的任务
            self.task_list = [t for t in self.task_list if t != self.current_task]
            self.current_task = None
            self.queue.task_done()

    def get_waiting_tasks(self):
        """获取等待中的任务列表（不从队列中移除）"""
        # 返回 task_list 中还在队列中的任务
        return [t for t in self.task_list if t != self.current_task]

    def clear(self):
        """清空队列"""
        while not self.queue.empty():
            self.queue.get_nowait()
        self.task_list = []
        self.repo_ids_in_queue = set()
        logger.info("🗑️ 任务队列已清空")


# 全局任务队列
task_queue = TaskQueue()

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

        # Fetch README from GitHub
        logger.info(f"📖 正在拉取 GitHub README...")
        readme_content = await github_monitor.fetch_repo_readme(repo_name)
        if readme_content:
            logger.info(f"✅ 成功获取 README ({len(readme_content)} 字符)")
        else:
            logger.info(f"ℹ️ 未找到 README，继续处理")

        # 2. RAG Refine
        logger.info(f"✍️ 正在生成初稿...")
        # Generate Draft with README
        draft = await rag_refiner.generate_draft(content, readme_content)

        # Generate AI Title
        logger.info(f"🏷️ 正在生成文档标题...")
        ai_title = await rag_refiner.generate_title(
            repo_name=repo_name,
            description=repo_data.get("description") or "",
            overview_content=content[:500]
        )
        
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

        # Save to local final_docs folder
        logger.info(f"💾 保存最终文档到本地...")
        final_docs_dir = "/www/wwwroot/mcp_deepwiki/final_docs"
        os.makedirs(final_docs_dir, exist_ok=True)
        safe_name = repo_name.replace("/", "_")
        final_doc_path = os.path.join(final_docs_dir, f"{safe_name}.md")
        with open(final_doc_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        logger.info(f"✅ 最终文档已保存: {final_doc_path}")

        # 3. Upload to Feishu
        logger.info(f"📤 正在上传到飞书知识库...")
        # Use AI-generated title with repo name
        title = f"{repo_name} - {ai_title}"
        logger.info(f"📌 文档标题：{title}")

        if not db_repo.feishu_doc_token:
            logger.info(f"🆕 创建新的飞书文档节点")
            doc_token = await feishu_service.create_node(title=title)
            if doc_token:
                db_repo.feishu_doc_token = doc_token
                db.commit()
        else:
            logger.info(f"📝 更新已有飞书文档")
            doc_token = db_repo.feishu_doc_token
            # 更新文档标题
            await feishu_service.update_node_title(doc_token, title)

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

async def queue_worker():
    """队列处理工作器 - 按顺序处理队列中的单个仓库任务"""
    logger.info("🔄 队列工作器已启动")

    while True:
        try:
            # 等待下一个任务
            task = await task_queue.get_next_task()
            task_queue.is_processing = True
            task_queue.current_task = task

            task_type = task["type"]
            repo_data = task["data"]
            repo_name = repo_data.get("full_name") or repo_data.get("repo_name", "Unknown")

            logger.info(f"📋 开始处理仓库: {repo_name} (任务类型: {task_type.value})")

            try:
                if task_type == TaskType.SYNC:
                    # 处理单个仓库的同步任务
                    db = SessionLocal()
                    try:
                        await process_repo_workflow(db, repo_data)
                    finally:
                        db.close()

                elif task_type == TaskType.REGENERATE:
                    # 处理重新生成任务
                    db = SessionLocal()
                    try:
                        await regenerate_repo_workflow_impl(db, repo_data["repo_id"])
                    finally:
                        db.close()

                logger.info(f"✅ 仓库处理完成: {repo_name}")

            except Exception as e:
                logger.error(f"❌ 仓库处理失败 [{repo_name}]: {e}")

            finally:
                task_queue.is_processing = False
                task_queue.mark_task_done()  # 使用新方法标记完成

        except Exception as e:
            logger.error(f"❌ 队列工作器错误: {e}")
            await asyncio.sleep(1)  # 避免快速循环

async def sync_task_impl(sync_all: bool = False, silent: bool = False):
    """同步任务的实际实现 - 不再查询数据库，直接使用传入的仓库列表"""
    # 注意：仓库列表已经在 sync_task 中查询并传入
    # 这里只是为了兼容旧的调用方式，如果 sync_task 直接调用（不带仓库列表）
    # 则需要在这里查询

    if not silent:
        logger.info(f"🔄 开始同步任务 (sync_all={sync_all})")

    db = SessionLocal()
    try:
        # Fetch new stars and pending repos (for backward compatibility)
        logger.info(f"⭐ 正在获取 GitHub 最新 star...")
        stars = await github_monitor.fetch_recent_stars(limit=10)
        logger.info(f"📦 发现 {len(stars)} 个新的 star 仓库")
        for star in stars:
            await process_repo_workflow(db, star)

        # Process pending/failed repositories from database (only FAILED and PENDING, not SKIPPED)
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

async def regenerate_repo_workflow_impl(db: Session, repo_id: str):
    """重新生成任务的实际实现"""
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    if not db_repo:
        logger.error(f"❌ 仓库不存在: {repo_id}")
        return

    repo_name = db_repo.repo_name
    logger.info(f"🔄 重新生成文档: {repo_name}")

    # Mark as processing
    db_repo.status = ProcessingStatus.PROCESSING
    db_repo.error_message = None
    db.commit()

    try:
        # Prepare repo data
        repo_data = {
            "id": db_repo.repo_id,
            "full_name": db_repo.repo_name,
            "html_url": db_repo.repo_url,
            "description": db_repo.description
        }

        # Run the full workflow
        await process_repo_workflow(db, repo_data)

        # If we have a doc_token and completed, clear the old content and regenerate
        if db_repo.feishu_doc_token and db_repo.status == ProcessingStatus.COMPLETED:
            logger.info(f"🗑️ 清空旧文档内容...")
            await feishu_service.clear_document_content(db_repo.feishu_doc_token)

            # Re-run the RAG and upload part
            safe_name = repo_name.replace("/", "_")
            base_dir = f"/www/wwwroot/mcp_deepwiki/output/{safe_name}"

            overview_files = glob.glob(os.path.join(base_dir, "*Overview.md"))
            if not overview_files:
                raise Exception("No Overview.md found")

            target_file = overview_files[0]
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Fetch README
            readme_content = await github_monitor.fetch_repo_readme(repo_name)

            # Generate Draft
            draft = await rag_refiner.generate_draft(content, readme_content)

            # Select documents
            candidate_files = [os.path.basename(p) for p in glob.glob(os.path.join(base_dir, "*.md")) if os.path.abspath(p) != os.path.abspath(target_file)]
            selected_files = await rag_refiner.select_documents_for_rag(draft, candidate_files)

            # Build Knowledge Base
            await rag_refiner.build_knowledge_base(base_dir, target_file, selected_files)

            # Final Expand
            final_content = await rag_refiner.expand_with_rag(draft)

            # Save to local final_docs folder
            logger.info(f"💾 保存最终文档到本地...")
            final_docs_dir = "/www/wwwroot/mcp_deepwiki/final_docs"
            os.makedirs(final_docs_dir, exist_ok=True)
            safe_name = repo_name.replace("/", "_")
            final_doc_path = os.path.join(final_docs_dir, f"{safe_name}.md")
            with open(final_doc_path, "w", encoding="utf-8") as f:
                f.write(final_content)
            logger.info(f"✅ 最终文档已保存: {final_doc_path}")

            # Generate AI Title
            ai_title = await rag_refiner.generate_title(
                repo_name=repo_name,
                description=repo_data.get("description") or "",
                overview_content=content[:500]
            )

            # Upload to Feishu (will append to empty doc)
            title = f"{repo_name} - {ai_title}"
            logger.info(f"📌 重新生成文档标题：{title}")

            # 更新文档标题
            await feishu_service.update_node_title(db_repo.feishu_doc_token, title)
            # 更新文档内容
            await feishu_service.update_document_content(db_repo.feishu_doc_token, final_content)

            # Send notification
            await feishu_service.send_card_notification(
                title=f"🔄 文档重新生成: {repo_name}",
                summary=repo_data.get("description") or "Documentation has been regenerated.",
                url=f"https://feishu.cn/docx/{db_repo.feishu_doc_token}"
            )
            await feishu_service.send_webhook_notification(repo_name, db_repo.feishu_doc_token)

            logger.info(f"✅ 重新生成完成: {repo_name}")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 重新生成失败 [{repo_name}]: {error_msg}")

        # Check if it's a cold repository error
        is_cold_repo = (
            "No Overview.md found" in error_msg or
            ("MCP fetch failed" in error_msg and "unindexed" in error_msg.lower()) or
            ("TaskGroup" in error_msg and "sub-exception" in error_msg)
        )

        if is_cold_repo:
            db_repo.status = ProcessingStatus.SKIPPED
            db_repo.error_message = f"冷门仓库：{error_msg}"
        else:
            db_repo.status = ProcessingStatus.FAILED
            db_repo.error_message = error_msg

        db.commit()

async def sync_task(sync_all: bool = False, silent: bool = False):
    """同步任务 - 将单个仓库逐个加入队列"""
    db = SessionLocal()
    try:
        # 1. Fetch new star repositories from GitHub
        stars = await github_monitor.fetch_recent_stars(limit=10)

        # 记录已处理的 GitHub star ID，避免重复
        processed_star_ids = set()

        # Collect all repos to process
        repos_to_process = []

        # 2. Process GitHub stars（只处理需要处理的）
        for star in stars:
            repo_id = str(star["id"])
            processed_star_ids.add(repo_id)  # 记录已处理

            db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()

            # 检查是否需要加入队列：
            # - 数据库中不存在 → 需要处理
            # - 数据库中是 PENDING/FAILED → 需要处理
            # - 数据库中是 COMPLETED/PROCESSING/SKIPPED → 不处理
            should_process = False
            if not db_repo:
                # 新仓库，创建记录
                db_repo = ProcessedRepo(
                    repo_id=repo_id,
                    repo_name=star["full_name"],
                    repo_url=star["html_url"],
                    description=star.get("description"),
                    status=ProcessingStatus.PENDING
                )
                db.add(db_repo)
                db.commit()
                should_process = True
            elif db_repo.status in [ProcessingStatus.PENDING, ProcessingStatus.FAILED]:
                # 待处理或失败的，需要重试
                should_process = True

            # 如果需要处理，且不在队列中，则加入
            if should_process and not task_queue.is_repo_in_queue(repo_id):
                repos_to_process.append(star)  # 直接使用 GitHub API 的原始格式

        # 3. 查询数据库中其他的 PENDING/FAILED 仓库（排除 GitHub star 列表中的）
        pending_repos = db.query(ProcessedRepo).filter(
            (ProcessedRepo.status == ProcessingStatus.PENDING) |
            (ProcessedRepo.status == ProcessingStatus.FAILED)
        ).all()

        # Add pending/failed repos（转换为 GitHub API 格式）
        for repo in pending_repos:
            # 跳过已经在 GitHub star 列表中处理过的
            if repo.repo_id in processed_star_ids:
                continue

            # 检查是否已在队列中
            if not task_queue.is_repo_in_queue(repo.repo_id):
                repos_to_process.append({
                    "id": repo.repo_id,  # 使用 "id" 而不是 "repo_id"
                    "full_name": repo.repo_name,  # 使用 "full_name" 而不是 "repo_name"
                    "html_url": repo.repo_url,  # 使用 "html_url" 而不是 "repo_url"
                    "description": repo.description,
                    "type": "retry"
                })

        # Add each repo as individual task to queue
        added_count = 0
        for repo_data in repos_to_process:
            await task_queue.add_repo_task(TaskType.SYNC, repo_data)
            added_count += 1

        if added_count > 0 and not silent:
            logger.info(f"📥 已加入 {added_count} 个仓库到队列")

    finally:
        db.close()

async def regenerate_repo_workflow(db: Session, repo_id: str):
    """重新生成任务 - 将单个仓库加入队列"""
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    if not db_repo:
        logger.error(f"❌ 仓库不存在: {repo_id}")
        return

    # 检查是否已在队列中
    if task_queue.is_repo_in_queue(repo_id):
        logger.info(f"ℹ️ 仓库已在队列中: {db_repo.repo_name}")
        return

    # 添加单个仓库任务
    await task_queue.add_repo_task(TaskType.REGENERATE, {
        "repo_id": repo_id,  # REGENERATE 任务特殊，需要 "repo_id" 字段
        "full_name": db_repo.repo_name,  # 统一使用 "full_name"
        "html_url": db_repo.repo_url,  # 统一使用 "html_url"
        "description": db_repo.description,
        "type": "regenerate"
    })

    logger.info(f"📥 重新生成任务已加入队列: {db_repo.repo_name}")

# Background Scheduler
async def scheduler_loop():
    while True:
        await asyncio.sleep(60)  # Run every 60 seconds (1 minute)
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

    # 启动队列工作器
    logger.info("🔄 启动任务队列工作器...")
    asyncio.create_task(queue_worker())

    logger.info("⏰ 启动后台调度器 (每60秒执行一次)")
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
async def retry_repo(repo_id: str, db: Session = Depends(get_db)):
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

    # Trigger a sync task in queue to process immediately
    await sync_task(False, True)

    logger.info(f"✅ 已将仓库 {db_repo.repo_name} 标记为待处理并加入队列")
    return {"status": "retrying"}

@app.post("/api/regenerate/{repo_id}")
async def regenerate_repo(repo_id: str, db: Session = Depends(get_db)):
    """Regenerate documentation for a completed repository"""
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    if not db_repo:
        logger.warning(f"⚠️ 重新生成失败：仓库 {repo_id} 不存在")
        return {"error": "Repository not found"}, 404

    if not db_repo.feishu_doc_token:
        logger.warning(f"⚠️ 仓库 {db_repo.repo_name} 尚未生成飞书文档，无法重新生成")
        return {"error": "No Feishu document found"}, 400

    logger.info(f"🔄 正在重新生成文档: {db_repo.repo_name}")
    # Add to queue instead of executing immediately
    await regenerate_repo_workflow(db, repo_id)

    return {"status": "regenerating"}

@app.post("/trigger")
async def trigger(sync_all: bool = False):
    logger.info(f"🎯 手动触发同步任务 (sync_all={sync_all})")
    await sync_task(sync_all)
    return {"status": "triggered"}

# Add API endpoint to check queue status
@app.get("/api/queue/status")
async def get_queue_status():
    """获取当前队列状态 - 返回单个仓库列表"""
    # Get waiting tasks from queue using the new method
    waiting_tasks = task_queue.get_waiting_tasks()

    # Build response - 单个仓库列表
    waiting_repos = []
    for task in waiting_tasks:
        data = task["data"]
        waiting_repos.append({
            "type": task["type"].value,
            "added_at": task["added_at"].isoformat(),
            "repo_id": data.get("repo_id") or str(data.get("id", "")),  # 兼容两种格式
            "repo_name": data.get("full_name") or data.get("repo_name", "Unknown"),
            "repo_type": data.get("type", "sync")
        })

    # Current task repo info
    current_repo = None
    if task_queue.current_task:
        data = task_queue.current_task["data"]
        current_repo = {
            "type": task_queue.current_task["type"].value,
            "added_at": task_queue.current_task["added_at"].isoformat(),
            "repo_id": data.get("repo_id") or str(data.get("id", "")),  # 兼容两种格式
            "repo_name": data.get("full_name") or data.get("repo_name", "Unknown"),
            "repo_type": data.get("type", "sync")
        }

    return {
        "queue_size": len(waiting_repos),
        "is_processing": task_queue.is_processing,
        "current_repo": current_repo,
        "waiting_repos": waiting_repos
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
