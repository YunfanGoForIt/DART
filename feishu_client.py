import lark_oapi as lark
from lark_oapi.api.wiki.v2 import *
from lark_oapi.api.docx.v1 import *
import logging
import httpx
from typing import List, Any, Optional

logger = logging.getLogger(__name__)

class FeishuService:
    def __init__(self, app_id: str, app_secret: str, space_id: str, webhook_url: Optional[str] = None):
        self.client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        self.space_id = space_id
        self.webhook_url = webhook_url

    async def create_node(self, title: str) -> Optional[str]:
        """Create a wiki node and return obj_token (document_id)"""
        try:
            request = CreateSpaceNodeRequest.builder() \
                .space_id(str(self.space_id)) \
                .request_body(Node.builder()
                    .obj_type("docx")
                    .node_type("origin")
                    .title(title)
                    .build()) \
                .build()

            response = self.client.wiki.v2.space_node.create(request)

            if not response.success():
                logger.error(f"Feishu create node error: {response.code} - {response.msg}")
                return None

            return response.data.node.obj_token
        except Exception as e:
            logger.error(f"Error creating Feishu node: {e}")
            return None

    async def update_node_title(self, node_token: str, title: str) -> bool:
        """Update wiki node title"""
        try:
            request = UpdateSpaceNodeRequest.builder() \
                .space_id(str(self.space_id)) \
                .node_token(str(node_token)) \
                .request_body(UpdateNodeRequestBody.builder()
                    .title(title)
                    .build()) \
                .build()

            response = self.client.wiki.v2.space_node.update(request)

            if not response.success():
                logger.error(f"Feishu update node title error: {response.code} - {response.msg}")
                return False

            logger.info(f"✅ Updated document title: {title}")
            return True
        except Exception as e:
            logger.error(f"Error updating Feishu node title: {e}")
            return False

    async def update_document_content(self, document_id: str, markdown_content: str):
        blocks = self._parse_markdown_to_blocks(markdown_content)
        
        try:
            # 1. Get root block
            list_request = ListDocumentBlockRequest.builder() \
                .document_id(document_id) \
                .build()
            
            list_response = self.client.docx.v1.document_block.list(list_request)
            if not list_response.success():
                logger.error(f"Failed to list blocks: {list_response.msg}")
                return

            root_block_id = list_response.data.items[0].block_id

            # 2. Create children blocks in batches
            batch_size = 50
            for i in range(0, len(blocks), batch_size):
                batch_blocks = blocks[i:i+batch_size]
                
                create_request = CreateDocumentBlockChildrenRequest.builder() \
                    .document_id(document_id) \
                    .block_id(root_block_id) \
                    .request_body(CreateDocumentBlockChildrenRequestBody.builder()
                        .index(-1)
                        .children(batch_blocks)
                        .build()) \
                    .build()
                
                create_response = self.client.docx.v1.document_block_children.create(create_request)
                if not create_response.success():
                    logger.error(f"Failed to create blocks batch: {create_response.msg}")

        except Exception as e:
            logger.error(f"Error updating document content: {e}")

    async def clear_document_content(self, document_id: str):
        """Clear all content from a document (delete all child blocks of root)"""
        try:
            # 1. Get all blocks
            list_request = ListDocumentBlockRequest.builder() \
                .document_id(document_id) \
                .build()

            list_response = self.client.docx.v1.document_block.list(list_request)
            if not list_response.success():
                logger.error(f"Failed to list blocks for clearing: {list_response.msg}")
                return False

            blocks = list_response.data.items
            if len(blocks) <= 1:
                # Only root block exists, nothing to clear
                return True

            root_block_id = blocks[0].block_id

            # 2. Delete all child blocks (except root)
            for block in blocks[1:]:
                delete_request = DeleteDocumentBlockRequest.builder() \
                    .document_id(document_id) \
                    .block_id(block.block_id) \
                    .build()

                delete_response = self.client.docx.v1.document_block.delete(delete_request)
                if not delete_response.success():
                    logger.warning(f"Failed to delete block {block.block_id}: {delete_response.msg}")

            logger.info(f"✅ Cleared document content: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing document content: {e}")
            return False

    def _parse_markdown_to_blocks(self, markdown: str) -> List[Any]:
        lines = markdown.split('\n')
        blocks = []
        in_code_block = False
        code_content = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('```'):
                if in_code_block:
                    in_code_block = False
                    blocks.append(self._create_code_block("\n".join(code_content)))
                    code_content = []
                else:
                    in_code_block = True
                continue
            
            if in_code_block:
                code_content.append(line)
                continue

            if not line:
                continue

            if line.startswith('# '):
                blocks.append(self._create_heading_block(line[2:], 1))
            elif line.startswith('## '):
                blocks.append(self._create_heading_block(line[3:], 2))
            elif line.startswith('### '):
                blocks.append(self._create_heading_block(line[4:], 3))
            elif line.startswith('- ') or line.startswith('* '):
                blocks.append(self._create_bullet_block(line[2:]))
            else:
                blocks.append(self._create_text_block(line))
                
        return blocks

    def _create_text_block(self, text: str):
        return {
            "block_type": 2,
            "text": {"elements": [{"text_run": {"content": text}}]}
        }

    def _create_heading_block(self, text: str, level: int):
        block_type = 2 + level
        return {
            "block_type": block_type,
            f"heading{level}": {"elements": [{"text_run": {"content": text}}]}
        }

    def _create_bullet_block(self, text: str):
        return {
            "block_type": 12,
            "bullet": {"elements": [{"text_run": {"content": text}}]}
        }

    def _create_code_block(self, content: str):
        return {
            "block_type": 14,
            "code": {
                "elements": [{"text_run": {"content": content}}],
                "style": {"language": 1}
            }
        }

    async def send_webhook_notification(self, repo_name: str, doc_token: str):
        """Send a simple webhook notification when a repo is finished."""
        if not self.webhook_url:
            return

        payload = {
            "msg_type": "text",
            "content": {
                "text": f"✅ 任务完成通知\n项目: {repo_name}\n飞书文档已生成: https://feishu.cn/docx/{doc_token}"
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(self.webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Error sending Feishu webhook notification: {e}")

    async def send_card_notification(self, title: str, summary: str, url: str):
        if not self.webhook_url:
            return

        card_content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": summary}},
                    {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "View Wiki"}, "url": url, "type": "primary"}]}
                ]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(self.webhook_url, json=card_content)
        except Exception as e:
            logger.error(f"Error sending Feishu notification: {e}")
