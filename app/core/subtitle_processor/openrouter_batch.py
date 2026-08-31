import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse

import requests

from app.config import CACHE_PATH
from app.core.utils.logger import setup_logger

logger = setup_logger("openrouter_batch")

_STATE_LOCK = threading.Lock()
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


@dataclass(frozen=True)
class OpenRouterBatchResult:
    batch_id: str
    data: Dict[str, Any]
    state_path: Path


class OpenRouterBatchRunner:
    """Submit or resume an OpenRouter batch backed by a persistent job record."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int,
        poll_interval: int = 10,
        state_dir: Path | None = None,
    ):
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.hostname != "openrouter.ai":
            raise ValueError(":batch 模型目前仅支持 OpenRouter API")

        self.batch_url = (
            f"{parsed_base_url.scheme}://{parsed_base_url.netloc}/api/beta/batches"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.state_dir = state_dir or Path(CACHE_PATH) / "openrouter_batches"

    def run(
        self,
        model: str,
        batch_requests: List[Dict[str, Any]],
        operation: str,
        is_running: Callable[[], bool],
        resume_identity: object | None = None,
    ) -> OpenRouterBatchResult:
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": model,
            "requests": batch_requests,
        }
        fingerprint = self._fingerprint(
            operation,
            {
                "model": model,
                "identity": resume_identity,
            }
            if resume_identity is not None
            else payload,
        )
        state_path = self.state_dir / f"{operation}-{fingerprint}.json"

        with _STATE_LOCK:
            state = self._load_state(state_path, fingerprint)
            resumed = state is not None
            if resumed:
                batch_id = state["batch_id"]
                logger.info("恢复 OpenRouter batch %s (%s)", batch_id, operation)
            else:
                batch_id = self._submit(payload)
                self._save_state(
                    state_path,
                    {
                        "version": 1,
                        "fingerprint": fingerprint,
                        "operation": operation,
                        "model": model,
                        "batch_id": batch_id,
                        "status": "submitted",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                logger.info(
                    "已提交 OpenRouter batch %s (%s)，包含 %d 个请求",
                    batch_id,
                    operation,
                    len(batch_requests),
                )

        not_found_retries = 0
        while is_running():
            response = requests.get(
                f"{self.batch_url}/{batch_id}",
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code == 404 and not_found_retries < 6:
                not_found_retries += 1
                logger.info(
                    "OpenRouter batch %s 尚未可查询，%d 秒后重试",
                    batch_id,
                    self.poll_interval,
                )
                time.sleep(self.poll_interval)
                continue
            if response.status_code == 404 and resumed:
                logger.warning("恢复的 OpenRouter batch %s 已失效，重新提交", batch_id)
                self._delete_state(state_path)
                return self.run(
                    model,
                    batch_requests,
                    operation,
                    is_running,
                    resume_identity,
                )

            response.raise_for_status()
            batch_data = response.json()
            status = batch_data.get("status")
            self._update_status(state_path, status)
            if status in _TERMINAL_STATUSES:
                break
            time.sleep(self.poll_interval)
        else:
            raise RuntimeError(f"OpenRouter batch {batch_id} 已停止")

        if status != "completed":
            self._delete_state(state_path)
            error = batch_data.get("error") or batch_data.get("errors") or status
            raise RuntimeError(f"OpenRouter batch {batch_id} 失败：{error}")

        return OpenRouterBatchResult(batch_id, batch_data, state_path)

    def mark_processed(self, result: OpenRouterBatchResult) -> None:
        """Remove the job record only after callers cache every batch result."""
        self._delete_state(result.state_path)

    def _submit(self, payload: Dict[str, Any]) -> str:
        response = requests.post(
            self.batch_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        batch_id = response.json().get("id")
        if not batch_id:
            raise RuntimeError("OpenRouter Batch API 未返回 batch id")
        return str(batch_id)

    def _fingerprint(self, operation: str, payload: Dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "batch_url": self.batch_url,
                "operation": operation,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load_state(
        self, state_path: Path, fingerprint: str
    ) -> Dict[str, Any] | None:
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("fingerprint") == fingerprint
                and state.get("batch_id")
            ):
                return state
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取 OpenRouter batch 状态失败，将重新提交：%s", exc)
        self._delete_state(state_path)
        return None

    def _save_state(self, state_path: Path, state: Dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_name(
            f"{state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temp_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp_path, state_path)

    def _update_status(self, state_path: Path, status: object) -> None:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = status
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state(state_path, state)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("更新 OpenRouter batch 状态失败：%s", exc)

    @staticmethod
    def _delete_state(state_path: Path) -> None:
        try:
            state_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("删除 OpenRouter batch 状态失败：%s", exc)


def parse_chat_completion_content(batch_result: Dict[str, Any]) -> str:
    if batch_result.get("error"):
        raise RuntimeError(f"OpenRouter batch 请求失败：{batch_result['error']}")

    response = batch_result.get("response", {})
    status_code = response.get("status_code")
    if status_code is not None and not 200 <= status_code < 300:
        raise RuntimeError(
            f"OpenRouter batch 请求失败 ({status_code})：{batch_result.get('error')}"
        )

    response_body = response.get("body", response)
    if not response_body:
        response_body = batch_result.get("body", {})
    try:
        return str(response_body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"无法解析 OpenRouter batch 响应：{batch_result}") from exc
