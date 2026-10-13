"""LLMクライアント。LLM_BACKEND=mock のときはNone(=ルール/テンプレートで動く)。"""
import json
import os
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol

from .errors import AIUnavailable, LLMError


class LLMClient(Protocol):
    async def complete_json(self, system: str, user: str, max_tokens: int = 1500) -> dict: ...


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSONが見つかりません")
    return json.loads(text[start : end + 1])


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        from anthropic import AsyncAnthropic  # 遅延import(mock時は不要)

        self._client = AsyncAnthropic(api_key=api_key, timeout=30.0, max_retries=1)
        self._model = model

    async def complete_json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # ネットワーク/レート制限/認証など
            raise LLMError(f"LLM呼び出しに失敗: {type(e).__name__}") from e
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return extract_json(text)


@lru_cache(maxsize=4)
def _build(backend: str, api_key: str, model: str) -> Optional[LLMClient]:
    if backend == "mock":
        return None
    if backend == "anthropic":
        if not api_key:
            raise AIUnavailable("ANTHROPIC_API_KEY が設定されていません。")
        return AnthropicLLM(api_key, model)
    raise AIUnavailable(f"未対応のLLM_BACKENDです: {backend}")


def get_llm() -> Optional[LLMClient]:
    """FastAPIのDepends用。環境変数を毎回読み、クライアントはキャッシュする。"""
    return _build(
        os.getenv("LLM_BACKEND", "mock").lower(),
        os.getenv("ANTHROPIC_API_KEY", ""),
        os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    )


async def ask_json(
    llm: LLMClient,
    system: str,
    user: str,
    check: Optional[Callable[[dict], Any]] = None,
    retries: int = 1,
    max_tokens: int = 1500,
) -> dict:
    """JSONを要求し、検証に失敗したら理由つきで1回だけ再試行する。"""
    last: Optional[Exception] = None
    for i in range(retries + 1):
        prompt = user if i == 0 else f"{user}\n\n前回の出力は不正でした({last})。指示を守り、JSONのみを返してください。"
        try:
            data = await llm.complete_json(system, prompt, max_tokens)
            if check:
                check(data)
            return data
        except (LLMError, ValueError, KeyError, TypeError) as e:
            last = e
    raise LLMError(str(last))
