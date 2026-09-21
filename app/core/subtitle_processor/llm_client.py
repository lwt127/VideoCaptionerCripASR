import asyncio
import atexit
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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


def find_agy_executable() -> str:
    """查找 Antigravity CLI (agy) 可执行文件路径"""
    env_bin = os.environ.get("ANTIGRAVITY_BIN") or os.environ.get("AGY_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin

    which_bin = shutil.which("agy") or shutil.which("agy.exe")
    if which_bin:
        return which_bin

    candidates = [
        Path.home() / ".gemini" / "bin" / "agy.exe",
        Path.home() / ".gemini" / "bin" / "agy",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise RuntimeError(
        "未找到 Antigravity CLI (agy) 可执行文件，请确保已安装并在 PATH 中或位于 ~/.gemini/bin/"
    )


def _get_subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def run_antigravity_completion(model: str, prompt: str, timeout: float = 60) -> str:
    agy_path = find_agy_executable()

    cmd = [
        agy_path,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
    ]
    if model and model != "auto":
        cmd.extend(["--model", model])

    msg = json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_get_subprocess_kwargs(),
    )

    try:
        stdout, stderr = proc.communicate(input=msg, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise TimeoutError(f"Antigravity CLI 请求超时 ({timeout}s)")

    response_content = None
    last_error = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("event") == "result":
                result = data.get("result", {})
                if result.get("status") == "SUCCESS":
                    response_content = result.get("response", "")
                else:
                    last_error = result.get("error") or "Antigravity CLI 执行失败"
        except Exception:
            continue

    if response_content is not None:
        return response_content

    error_msg = last_error or stderr.strip() or "Antigravity CLI 未返回有效响应"
    raise RuntimeError(f"Antigravity CLI 错误: {error_msg}")


def list_antigravity_models() -> list[str]:
    agy_path = find_agy_executable()
    proc = subprocess.run(
        [agy_path, "models"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        **_get_subprocess_kwargs(),
    )
    models = ["auto"]
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Fetching"):
            continue
        parts = line.split("\t")
        model_id = parts[0].strip()
        if model_id and model_id not in models:
            models.append(model_id)

    return models


def test_antigravity(model: str) -> tuple[bool, str, list[str]]:
    try:
        models = list_antigravity_models()
    except Exception as exc:
        return False, str(exc), []

    if model != "auto" and model not in models:
        return False, f"Antigravity 模型不可用: {model}", models

    try:
        run_antigravity_completion(model, "ping", timeout=20)
    except Exception as exc:
        return False, f"Antigravity CLI 测试失败: {exc}", models

    return True, "Antigravity CLI 连接成功", models


class AntigravityChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float = 60,
        **_: Any,
    ) -> Any:
        if model.endswith(":batch"):
            raise ValueError(":batch 模型不支持 Antigravity CLI")

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
        user_prompt = "\n\n".join(user_messages)

        full_prompt = (
            f"System Instructions:\n{system_prompt}\n\nUser:\n{user_prompt}"
            if system_prompt
            else user_prompt
        )

        content = run_antigravity_completion(
            model=model,
            prompt=full_prompt,
            timeout=timeout,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class AntigravityClientAdapter:
    def __init__(self):
        self.chat = SimpleNamespace(completions=AntigravityChatCompletions())


def create_llm_client(
    llm_service: Optional[LLMServiceEnum],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    if llm_service == LLMServiceEnum.GITHUB_COPILOT:
        return CopilotClientAdapter()
    if llm_service == LLMServiceEnum.ANTIGRAVITY:
        return AntigravityClientAdapter()
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