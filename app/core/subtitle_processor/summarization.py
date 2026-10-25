import os
from typing import Optional

from app.core.entities import LLMServiceEnum
from app.core.subtitle_processor.llm_client import create_llm_client

from ..utils import json_repair
from ..utils.logger import setup_logger
from .prompt import SUMMARIZER_PROMPT

logger = setup_logger("subtitle_summarizer")


class SubtitleSummarizer:
    def __init__(
        self, model: str, llm_service: Optional[LLMServiceEnum] = None
    ) -> None:
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")

        self.model = model
        self.client = create_llm_client(llm_service, base_url, api_key)

    def summarize(self, subtitle_content: str) -> str:
        logger.info("开始摘要化字幕内容")
        try:
            subtitle_content = subtitle_content[:3000]
            response = self.client.chat.completions.create(
                model=self.model,
                stream=False,
                messages=[
                    {"role": "system", "content": SUMMARIZER_PROMPT},
                    {
                        "role": "user",
                        "content": f"summarize the video content:\n{subtitle_content}",
                    },
                ],
            )
            return str(json_repair.loads(response.choices[0].message.content))
        except Exception as e:
            logger.exception(f"摘要化字幕内容失败: {e}")
            return ""
