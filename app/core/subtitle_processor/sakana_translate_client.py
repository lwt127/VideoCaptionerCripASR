# app/core/subtitle_processor/sakana_translate_client.py
"""
非官方客户端：对接 https://chat.sakana.ai/translate 网页版的后端接口。

工作原理
--------
该网页本身没有公开的、需要申请密钥的 REST API。它的前端通过以下步骤工作：

1. 使用 Firebase 匿名登录（Anonymous Auth）获取一个 ``idToken``。
2. 用该 ``idToken`` 调用 ``POST /api/auth/login`` 换取会话 Cookie
   （``sakana-chat``），后续请求携带该 Cookie 即被视为已登录的"访客"。
3. 调用 ``POST /translate/api/translate``（SSE 流式响应）获取翻译结果。
   请求体为 ``{"text", "sourceLang", "targetLang", "nativeLang"}``。

限制
----
* ``sourceLang`` / ``targetLang`` 只能是 ``en`` / ``ja`` / ``zh`` / ``zh-Hant`` 之一，
  不支持 ``auto`` 自动检测。
* 单次翻译文本上限约 2000 字符（与网页输入框限制一致）。
* 匿名访客账号有速率限制：约每分钟 10 次、每日 15 次文本翻译
  （以官方接口返回为准，可能随时调整）。命中限制时会返回 HTTP 429。

由于这是通过分析网页请求得到的非官方接口，建议仅用于个人轻量使用
（例如翻译少量字幕行、临时性翻译需求），不要用于大规模高频调用，
以免对该服务造成负担或触发更严格的风控。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from app.core.utils.logger import setup_logger

logger = setup_logger("sakana_translate_client")

# Firebase 项目信息（从网页前端的公开请求中提取，非敏感密钥）
_FIREBASE_API_KEY = "AIzaSyBIJuyUokxGiETY0Nu3hQNC1dMadHyf_I4"
_FIREBASE_TENANT_ID = "sakana-talk-prd-pvl72"
_SIGNUP_URL = (
    f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={_FIREBASE_API_KEY}"
)
_LOGIN_URL = "https://chat.sakana.ai/api/auth/login"
_TRANSLATE_URL = "https://chat.sakana.ai/translate/api/translate"
_REFERER = "https://chat.sakana.ai/translate"
_ORIGIN = "https://chat.sakana.ai"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Sakana Translate 网页版仅支持这四种语言代码
SUPPORTED_LANGS = {"en", "ja", "zh", "zh-Hant"}

MAX_TEXT_LENGTH = 2000


class SakanaTranslateError(RuntimeError):
    """Sakana Translate 接口调用异常"""


@dataclass
class SakanaTranslation:
    variant: str  # "casual" 或 "polite"
    text: str
    notes: List[str]


class SakanaTranslateClient:
    """线程安全的 Sakana Translate 客户端，自动管理匿名会话及限速重试。"""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()
        self._base_headers = {
            "User-Agent": _USER_AGENT,
            "Referer": _REFERER,
        }

    # ------------------------------------------------------------------
    # 会话管理：匿名登录 + 换取 Cookie
    # ------------------------------------------------------------------
    def _create_session(self) -> requests.Session:
        session = requests.Session()

        signup_resp = session.post(
            _SIGNUP_URL,
            headers={**self._base_headers, "Content-Type": "application/json"},
            json={"returnSecureToken": True, "tenantId": _FIREBASE_TENANT_ID},
            timeout=self.timeout,
        )
        signup_resp.raise_for_status()
        id_token = signup_resp.json().get("idToken")
        if not id_token:
            raise SakanaTranslateError("Firebase匿名登录失败：响应中缺少idToken")

        login_resp = session.post(
            _LOGIN_URL,
            headers={**self._base_headers, "Origin": _ORIGIN},
            files={"idToken": (None, id_token)},
            timeout=self.timeout,
        )
        if login_resp.status_code != 200:
            raise SakanaTranslateError(
                f"Sakana 会话建立失败：HTTP {login_resp.status_code} {login_resp.text[:200]}"
            )
        logger.debug("已创建新的 Sakana 匿名会话")
        return session

    def _get_session(self, force_new: bool = False) -> requests.Session:
        with self._lock:
            if force_new or self._session is None:
                self._session = self._create_session()
            return self._session

    # ------------------------------------------------------------------
    # SSE 响应解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sse_done_event(resp: requests.Response) -> Dict[str, Any]:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "done":
                return event
        raise SakanaTranslateError("Sakana翻译接口未返回完整结果（缺少done事件）")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        native_lang: str = "en",
        variant: str = "casual",
    ) -> str:
        """
        翻译一段文本。

        :param text: 待翻译文本（超过 ``MAX_TEXT_LENGTH`` 会被截断）
        :param source_lang: 源语言，取值范围见 :data:`SUPPORTED_LANGS`
        :param target_lang: 目标语言，取值范围见 :data:`SUPPORTED_LANGS`
        :param native_lang: 用于生成翻译注释说明的界面语言，不影响译文本身
        :param variant: ``"casual"``（口语化）或 ``"polite"``（礼貌/正式）
        :return: 译文文本
        """
        if source_lang not in SUPPORTED_LANGS:
            raise SakanaTranslateError(
                f"不支持的源语言：{source_lang}（仅支持 {sorted(SUPPORTED_LANGS)}）"
            )
        if target_lang not in SUPPORTED_LANGS:
            raise SakanaTranslateError(
                f"不支持的目标语言：{target_lang}（仅支持 {sorted(SUPPORTED_LANGS)}）"
            )

        text = text[:MAX_TEXT_LENGTH]
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            session = self._get_session()
            try:
                resp = session.post(
                    _TRANSLATE_URL,
                    headers={**self._base_headers, "Content-Type": "application/json"},
                    json={
                        "text": text,
                        "sourceLang": source_lang,
                        "targetLang": target_lang,
                        "nativeLang": native_lang,
                    },
                    timeout=self.timeout,
                    stream=True,
                )

                if resp.status_code == 401:
                    last_error = SakanaTranslateError("会话已过期 (401)")
                    logger.warning("Sakana 会话已过期，正在重新登录...")
                    self._get_session(force_new=True)
                    continue

                if resp.status_code == 429:
                    detail: Dict[str, Any] = {}
                    try:
                        detail = resp.json()
                    except ValueError:
                        pass
                    context = detail.get("context") or {}
                    retry_after = min(int(context.get("retry_after", 5)), 60)
                    limit_type = context.get("limit_type", "unknown")
                    logger.warning(
                        f"Sakana翻译触发限速（{limit_type}），等待{retry_after}秒后切换新会话重试"
                    )
                    time.sleep(retry_after)
                    # 使用新的匿名会话重试（新账号拥有独立的分钟/每日额度）
                    self._get_session(force_new=True)
                    last_error = SakanaTranslateError(f"限速：{limit_type}")
                    continue

                resp.raise_for_status()
                done_event = self._parse_sse_done_event(resp)
                translations = done_event.get("translations") or []
                if not translations:
                    raise SakanaTranslateError("Sakana翻译结果为空")

                chosen = next(
                    (t for t in translations if t.get("type") == variant),
                    translations[0],
                )
                return chosen["text"]
            except (requests.RequestException, SakanaTranslateError) as e:
                last_error = e
                logger.warning(f"Sakana翻译第{attempt + 1}次尝试失败：{e}")
                continue

        raise SakanaTranslateError(
            f"Sakana翻译失败，已重试{self.max_retries}次：{last_error}"
        )
