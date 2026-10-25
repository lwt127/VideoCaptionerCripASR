import asyncio
import atexit
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Optional

from openai import OpenAI

from app.config import CACHE_PATH
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


# ------------------- Antigravity CLI (agy) -------------------

_AGY_AGENT_NAME = "vc-llm"
_AGY_AGENT_MD = """---
name: vc-llm
description: Stateless text and JSON completion engine for VideoCaptioner
enable_write_tools: false
enable_mcp_tools: false
enable_subagent_tools: false
---
You are a stateless LLM completion engine for VideoCaptioner.
CRITICAL RULES:
1. NEVER invoke any tools (do not read files, list directories, run commands, search the web, or ask questions).
2. Follow the instructions inside <SYSTEM_INSTRUCTIONS> strictly.
3. Process the data inside <USER_INPUT> and output ONLY the final response text or JSON without extra conversational filler.
"""

_AGY_MAX_CONCURRENCY = max(1, int(os.environ.get("AGY_MAX_CONCURRENCY", "4")))
_agy_semaphore = threading.BoundedSemaphore(_AGY_MAX_CONCURRENCY)
_agy_workspace_lock = threading.Lock()
_agy_workspace_dir: Optional[str] = None

_agy_models_lock = threading.Lock()
_agy_models_cache: tuple[float, list[str]] = (0.0, [])
_AGY_MODELS_CACHE_TTL = 60.0


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


def _ensure_agy_workspace() -> str:
    """準備隔離嘅空工作目錄同埋極簡無工具 Agent 定義，大幅減少每次請求嘅 Token 開銷"""
    global _agy_workspace_dir
    with _agy_workspace_lock:
        if _agy_workspace_dir and os.path.isdir(_agy_workspace_dir):
            agent_file = (
                Path(_agy_workspace_dir)
                / ".agents"
                / "agents"
                / _AGY_AGENT_NAME
                / "agent.md"
            )
            if agent_file.is_file():
                return _agy_workspace_dir

        base_dir = (
            Path(CACHE_PATH) / "antigravity_workspace"
            if CACHE_PATH
            else Path(tempfile.gettempdir()) / "vc_antigravity_workspace"
        )
        agent_dir = base_dir / ".agents" / "agents" / _AGY_AGENT_NAME
        agent_dir.mkdir(parents=True, exist_ok=True)
        agent_file = agent_dir / "agent.md"
        if not agent_file.is_file() or agent_file.read_text(encoding="utf-8") != _AGY_AGENT_MD:
            agent_file.write_text(_AGY_AGENT_MD, encoding="utf-8")

        _agy_workspace_dir = str(base_dir)
        return _agy_workspace_dir


def _cleanup_agy_conversation(conversation_id: Optional[str]) -> None:
    """清理每次臨時請求產生嘅 Brain 對話紀錄資料夾，避免污染用戶嘅 Antigravity History"""
    if not conversation_id or "/" in conversation_id or "\\" in conversation_id or ".." in conversation_id:
        return
    try:
        brain_dir = Path.home() / ".gemini" / "antigravity" / "brain" / conversation_id
        if brain_dir.is_dir():
            shutil.rmtree(brain_dir, ignore_errors=True)
    except Exception:
        pass


def _get_subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _run_single_antigravity_turn(
    agy_path: str,
    workspace_dir: str,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    cmd = [
        agy_path,
        "--agent",
        _AGY_AGENT_NAME,
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
        cwd=workspace_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_get_subprocess_kwargs(),
    )

    conversation_id: Optional[str] = None
    try:
        try:
            stdout, stderr = proc.communicate(input=msg, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, _ = proc.communicate(timeout=2)
                for line in (stdout or "").splitlines():
                    try:
                        data = json.loads(line.strip())
                        conversation_id = conversation_id or data.get("conversation_id")
                    except Exception:
                        pass
            except Exception:
                pass
            raise TimeoutError(f"Antigravity CLI 请求超时 ({timeout}s)")

        response_content: Optional[str] = None
        last_error: Optional[str] = None

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not conversation_id:
                    conversation_id = data.get("conversation_id")
                if data.get("event") == "result":
                    result = data.get("result", {})
                    if not conversation_id:
                        conversation_id = result.get("conversation_id")
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
    finally:
        _cleanup_agy_conversation(conversation_id)


def run_antigravity_completion(
    model: str,
    prompt: str,
    timeout: float = 60,
    max_retries: int = 2,
) -> str:
    agy_path = find_agy_executable()
    workspace_dir = _ensure_agy_workspace()

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        with _agy_semaphore:
            try:
                return _run_single_antigravity_turn(
                    agy_path=agy_path,
                    workspace_dir=workspace_dir,
                    model=model,
                    prompt=prompt,
                    timeout=timeout,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    raise
        time.sleep(min(1.5 * (attempt + 1), 5.0))

    assert last_exc is not None
    raise last_exc


def list_antigravity_models(use_cache: bool = True) -> list[str]:
    global _agy_models_cache
    now = time.monotonic()
    if use_cache:
        with _agy_models_lock:
            cached_at, cached_models = _agy_models_cache
            if cached_models and (now - cached_at) < _AGY_MODELS_CACHE_TTL:
                return list(cached_models)

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
    if proc.returncode != 0:
        err_detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"获取 Antigravity 模型列表失败: {err_detail or f'exit code {proc.returncode}'}"
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

    if len(models) <= 1:
        raise RuntimeError("Antigravity CLI 未返回可用模型，请确认已登录 `agy`")

    with _agy_models_lock:
        _agy_models_cache = (time.monotonic(), list(models))

    return models


def test_antigravity(model: str, use_cache: bool = True) -> tuple[bool, str, list[str]]:
    try:
        models = list_antigravity_models(use_cache=use_cache)
    except Exception as exc:
        return False, str(exc), []

    if model and model != "auto" and model not in models:
        return False, f"Antigravity 模型不可用: {model}", models

    return True, "Antigravity CLI 已就绪", models


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

        system_parts: list[str] = []
        conversation_parts: list[str] = []

        non_system_messages = [m for m in messages if m.get("role") != "system"]
        has_assistant_turns = any(
            m.get("role") == "assistant" for m in non_system_messages
        )

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "system":
                if content:
                    system_parts.append(content)
            elif has_assistant_turns:
                role_tag = "ASSISTANT" if role == "assistant" else "USER"
                conversation_parts.append(f"[{role_tag}]\n{content}")
            else:
                conversation_parts.append(content)

        system_prompt = "\n\n".join(system_parts).strip()
        user_prompt = "\n\n".join(conversation_parts).strip()

        if system_prompt:
            full_prompt = (
                f"<SYSTEM_INSTRUCTIONS>\n{system_prompt}\n</SYSTEM_INSTRUCTIONS>\n\n"
                f"<USER_INPUT>\n{user_prompt}\n</USER_INPUT>"
            )
        else:
            full_prompt = user_prompt

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
