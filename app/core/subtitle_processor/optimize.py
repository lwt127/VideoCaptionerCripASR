import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import retry
from openai import OpenAI

from app.config import CACHE_PATH
from app.core.bk_asr.asr_data import ASRData, ASRDataSeg
from app.core.storage.cache_manager import CacheManager
from app.core.utils import json_repair
from app.core.subtitle_processor.alignment import SubtitleAligner
from app.core.subtitle_processor.openrouter_batch import (
    OpenRouterBatchRunner,
    parse_chat_completion_content,
)
from app.core.subtitle_processor.prompt import OPTIMIZER_PROMPT
from app.core.utils.logger import setup_logger

logger = setup_logger("subtitle_optimizer")


class SubtitleOptimizer:
    """字幕优化器,支持缓存功能"""

    BATCH_POLL_INTERVAL = 10

    def __init__(
        self,
        thread_num: int = 5,
        batch_num: int = 10,
        model: str = "gpt-4o-mini",
        custom_prompt: str = "",
        temperature: float = 0.7,
        timeout: int = 60,
        retry_times: int = 1,
        update_callback: Optional[Callable] = None,
    ):
        self._init_client()
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.model = model
        self.custom_prompt = custom_prompt
        self.temperature = temperature
        self.timeout = timeout
        self.retry_times = retry_times
        self.is_running = True
        self.update_callback = update_callback
        self._init_thread_pool()
        self.cache_manager = CacheManager(str(CACHE_PATH))

    def _init_client(self):
        """初始化OpenAI客户端"""
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not (base_url and api_key):
            raise ValueError("环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY 必须设置")

        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _init_thread_pool(self):
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        import atexit

        atexit.register(self.stop)

    def optimize_subtitle(self, subtitle_data: Union[str, ASRData]) -> ASRData:
        """优化字幕文件"""
        try:
            # 读取字幕文件
            if isinstance(subtitle_data, str):
                asr_data = ASRData.from_subtitle_file(subtitle_data)
            else:
                asr_data = subtitle_data

            # 将ASRData转换为字典格式
            subtitle_dict = {
                str(i): seg.text for i, seg in enumerate(asr_data.segments, 1)
            }

            # 分批处理字幕
            chunks = self._split_chunks(subtitle_dict)

            # 多线程优化
            optimized_dict = self._parallel_optimize(chunks)

            # 创建新的ASRDataSeg列表
            new_segments = self._create_segments(asr_data.segments, optimized_dict)

            return ASRData(new_segments)
        except Exception as e:
            logger.error(f"优化失败：{str(e)}")
            raise RuntimeError(f"优化失败：{str(e)}")

    def _split_chunks(self, subtitle_dict: Dict[str, str]) -> List[Dict[str, str]]:
        """将字幕分割成块"""
        items = list(subtitle_dict.items())
        return [
            dict(items[i : i + self.batch_num])
            for i in range(0, len(items), self.batch_num)
        ]

    def _parallel_optimize(self, chunks: List[Dict[str, str]]) -> Dict[str, str]:
        """并行优化所有块"""
        if self.model.endswith(":batch"):
            try:
                return self._optimize_openrouter_batch(chunks)
            except Exception as e:
                logger.error(f"Batch 优化失败，保留原字幕：{str(e)}")
                return {
                    key: value for chunk in chunks for key, value in chunk.items()
                }

        future_chunks = {}
        optimized_dict = {}

        for chunk in chunks:
            if not self.executor:
                raise ValueError("线程池未初始化")
            future = self.executor.submit(self._safe_optimize_chunk, chunk)
            future_chunks[future] = chunk

        for future in as_completed(future_chunks):
            if not self.is_running:
                logger.info("优化器已停止运行，退出优化")
                break
            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                logger.error(f"优化块失败：{str(e)}")
                # 对于失败的块，保留原文
                for k, v in future_chunks[future].items():
                    optimized_dict[k] = v

        return optimized_dict

    def _optimize_openrouter_batch(
        self, chunks: List[Dict[str, str]]
    ) -> Dict[str, str]:
        optimized_dict: Dict[str, str] = {}
        pending_chunks = {}
        batch_requests = []
        base_model = self.model.removesuffix(":batch")

        for chunk_index, chunk in enumerate(chunks):
            user_prompt = self._get_user_prompt(chunk)
            cache_params = {
                "temperature": self.temperature,
                "model": self.model,
            }
            cache_key = f"{len(OPTIMIZER_PROMPT)}_{user_prompt}"
            cache_result = self.cache_manager.get_llm_result(
                cache_key, self.model, **cache_params
            )
            if cache_result:
                try:
                    optimized_dict.update(
                        self._parse_optimized_result(
                            json.loads(cache_result), "缓存"
                        )
                    )
                    continue
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"优化缓存格式无效，重新调用API: {str(e)}")

            custom_id = f"optimizer-chunk-{chunk_index}"
            pending_chunks[custom_id] = (
                chunk,
                cache_key,
                cache_params,
            )
            batch_requests.append(
                {
                    "custom_id": custom_id,
                    "body": {
                        "model": base_model,
                        "messages": [
                            {"role": "system", "content": OPTIMIZER_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.temperature,
                    },
                }
            )

        if not batch_requests:
            return optimized_dict

        runner = OpenRouterBatchRunner(
            self.base_url,
            self.api_key,
            self.timeout,
            self.BATCH_POLL_INTERVAL,
            getattr(self, "batch_state_dir", None),
        )
        batch = runner.run(
            base_model,
            batch_requests,
            "optimization",
            lambda: self.is_running,
            {
                "model": self.model,
                "chunks": chunks,
                "custom_prompt": self.custom_prompt,
                "temperature": self.temperature,
            },
        )
        results_by_id = {
            result.get("custom_id"): result
            for result in batch.data.get("results", [])
        }

        for custom_id, (chunk, cache_key, cache_params) in pending_chunks.items():
            try:
                batch_result = results_by_id.get(custom_id)
                if not batch_result:
                    raise RuntimeError(
                        f"OpenRouter batch 缺少结果：{custom_id}"
                    )
                result = self._parse_optimized_result(
                    json_repair.loads(
                        parse_chat_completion_content(batch_result)
                    ),
                    "Batch API",
                )
                aligned_result = self._repair_subtitle(chunk, result)
                self.cache_manager.set_llm_result(
                    cache_key,
                    json.dumps(aligned_result, ensure_ascii=False),
                    self.model,
                    **cache_params,
                )
                optimized_dict.update(aligned_result)
                if self.update_callback:
                    self.update_callback(aligned_result)
            except Exception as e:
                logger.error(f"Batch 优化块失败 {custom_id}：{str(e)}")
                optimized_dict.update(chunk)

        runner.mark_processed(batch)
        return optimized_dict

    def _safe_optimize_chunk(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """安全的优化块，包含重试逻辑"""
        for i in range(self.retry_times):
            try:
                return self._optimize_chunk(chunk)
            except Exception as e:
                if i == self.retry_times - 1:
                    raise
                logger.warning(f"优化重试 {i+1}/{self.retry_times}: {str(e)}")
        return chunk

    def _optimize_chunk(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """优化字幕块"""
        logger.info(
            f"[+]正在优化字幕：{next(iter(subtitle_chunk))} - {next(reversed(subtitle_chunk))}"
        )
        user_prompt = self._get_user_prompt(subtitle_chunk)

        # 检查缓存
        cache_params = {
            "temperature": self.temperature,
            "model": self.model,
        }
        # 构建缓存key
        cache_key = f"{len(OPTIMIZER_PROMPT)}_{user_prompt}"
        cache_result = self.cache_manager.get_llm_result(
            cache_key, self.model, **cache_params
        )

        if cache_result:
            try:
                cached_result = self._parse_optimized_result(
                    json.loads(cache_result), "缓存"
                )
                logger.info("使用缓存的优化结果")
                return cached_result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"优化缓存格式无效，重新调用API: {str(e)}")

        # 构建提示词
        messages = [
            {"role": "system", "content": OPTIMIZER_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # 调用API优化
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            timeout=self.timeout,
        )

        # 解析结果
        result = self._parse_optimized_result(
            json_repair.loads(response.choices[0].message.content),  # type: ignore
            "API",
        )

        # 修复字幕对齐
        aligned_result = self._repair_subtitle(subtitle_chunk, result)

        # 保存到缓存
        self.cache_manager.set_llm_result(
            cache_key,
            json.dumps(aligned_result, ensure_ascii=False),
            self.model,
            **cache_params,
        )

        if self.update_callback:
            self.update_callback(aligned_result)

        return aligned_result

    def _get_user_prompt(self, subtitle_chunk: Dict[str, str]) -> str:
        user_prompt = f"Correct the following subtitles. Keep the original language, do not translate:\n<input_subtitle>{str(subtitle_chunk)}</input_subtitle>"
        if self.custom_prompt:
            user_prompt += (
                f"\nReference content:\n<prompt>{self.custom_prompt}</prompt>"
            )
        return user_prompt

    @staticmethod
    def _parse_optimized_result(result: object, source: str) -> Dict[str, str]:
        """验证优化器返回的字幕映射。"""
        if not isinstance(result, dict):
            raise ValueError(
                f"{source}返回格式无效：期望JSON对象，实际为{type(result).__name__}"
            )
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in result.items()):
            raise ValueError(f"{source}返回格式无效：JSON对象的键和值必须都是字符串")
        return result

    @staticmethod
    def _repair_subtitle(
        original: Dict[str, str], optimized: Dict[str, str]
    ) -> Dict[str, str]:
        """修复字幕对齐问题"""
        aligner = SubtitleAligner()
        original_list = list(original.values())
        optimized_list = list(optimized.values())

        aligned_source, aligned_target = aligner.align_texts(
            original_list, optimized_list
        )

        if len(aligned_source) != len(aligned_target):
            raise ValueError("对齐后字幕长度不一致")

        # 构建对齐后的字典
        start_id = next(iter(original.keys()))
        return {str(int(start_id) + i): text for i, text in enumerate(aligned_target)}

    @staticmethod
    def _create_segments(
        original_segments: List[ASRDataSeg],
        optimized_dict: Dict[str, str],
    ) -> List[ASRDataSeg]:
        """创建新的字幕段"""
        return [
            ASRDataSeg(
                text=optimized_dict.get(str(i), seg.text),
                start_time=seg.start_time,
                end_time=seg.end_time,
            )
            for i, seg in enumerate(original_segments, 1)
        ]

    def stop(self):
        """停止优化器"""
        if not self.is_running:
            return

        logger.info("正在停止优化器...")
        self.is_running = False
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error(f"关闭线程池时出错：{str(e)}")
            finally:
                self.executor = None
