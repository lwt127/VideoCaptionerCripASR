import asyncio
import atexit
import json
import threading
from types import SimpleNamespace
from typing import Any, Optional

from openai import OpenAI

from app.core.entities import LLMServiceEnum


class CopilotSDKBridge:
    """Expose the async Copilot SDK through the app's synchronous workers."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.client = None
        self._start_lock = asyncio.Lock()

    async def _get_client(self):
        async with self._start_lock:
            if self.client is None:
                try:
                    from copilot import CopilotClient
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装 GitHub Copilot SDK，请重新安装 requirements.txt"
                    ) from exc
                self.client = CopilotClient(use_logged_in_user=True)
                await self.client.start()
        return self.client

    async def _complete(
        self, model: str, system_prompt: str, user_prompt: str, timeout: float
    ) -> str:
        client = await self._get_client()
        session_options = dict(
            model=model,
            available_tools=[],
            infinite_sessions={"enabled": False},
            memory={"enabled": False},
            enable_session_store=False,
            enable_skills=False,
            enable_config_discovery=False,
            streaming=False,
        )
        if system_prompt:
            session_options["system_message"] = {
                "mode": "replace",
                "content": system_prompt,
            }
        session = await client.create_session(**session_options)
        try:
            response = await session.send_and_wait(user_prompt, timeout=timeout)
            content = getattr(getattr(response, "data", None), "content", None)
            if not content:
                raise RuntimeError("GitHub Copilot 未返回内容")
            return content
        finally:
            await session.disconnect()

    def complete(
        self, model: str, system_prompt: str, user_prompt: str, timeout: float
    ) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._complete(model, system_prompt, user_prompt, timeout), self.loop
        )
        return future.result(timeout=timeout + 30)

    async def _list_models(self) -> list[str]:
        client = await self._get_client()
        return [model.id for model in await client.list_models()]

    def list_models(self) -> list[str]:
        future = asyncio.run_coroutine_threadsafe(self._list_models(), self.loop)
        return future.result(timeout=60)

    def stop(self):
        if self.loop.is_closed():
            return
        if self.client is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.client.stop(), self.loop
                )
                future.result(timeout=15)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)


_copilot_bridge: Optional[CopilotSDKBridge] = None
_copilot_bridge_lock = threading.Lock()


def get_copilot_bridge() -> CopilotSDKBridge:
    global _copilot_bridge
    with _copilot_bridge_lock:
        if _copilot_bridge is None:
            _copilot_bridge = CopilotSDKBridge()
            atexit.register(_copilot_bridge.stop)
        return _copilot_bridge


class CopilotChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float = 60,
        **_: Any,
    ) -> Any:
        if model.endswith(":batch"):
            raise ValueError(":batch 模型不支持 GitHub Copilot")

        system_prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        user_messages = []
        for message in messages:
            if message.get("role") == "system":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            user_messages.append(content)
        content = get_copilot_bridge().complete(
            model, system_prompt, "\n".join(user_messages), timeout
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class CopilotClientAdapter:
    def __init__(self):
        self.chat = SimpleNamespace(completions=CopilotChatCompletions())


def create_llm_client(
    llm_service: Optional[LLMServiceEnum],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    if llm_service == LLMServiceEnum.GITHUB_COPILOT:
        return CopilotClientAdapter()
    if not (base_url and api_key):
        raise ValueError("环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY 必须设置")
    return OpenAI(base_url=base_url, api_key=api_key)


def test_copilot(model: str) -> tuple[bool, str, list[str]]:
    try:
        models = get_copilot_bridge().list_models()
    except Exception as exc:
        if "not authenticated" in str(exc).lower():
            raise RuntimeError(
                "GitHub Copilot 未登录，请先使用 GitHub Copilot CLI 登录"
            ) from exc
        raise
    if model != "auto" and model not in models:
        return False, f"Copilot 模型不可用: {model}", models
    return True, "GitHub Copilot 已登录", models