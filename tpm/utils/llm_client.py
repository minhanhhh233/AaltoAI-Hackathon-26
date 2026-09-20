"""
LLM Client Abstraction
-----------------------
A minimal, provider-agnostic interface for a chat-capable LLM backend,
so Layer 4 (layer4_llm.py) depends only on `LLMClient`, never on a
specific provider's SDK. Swapping backends — OpenAI's API today, a
different provider or a locally-hosted model tomorrow — is a config
change, not a code change.

OpenAICompatibleClient covers more ground than its name suggests: any
backend that implements OpenAI's Chat Completions wire format (real
OpenAI, and most self-hosted model servers — Ollama, vLLM, LM Studio,
llama.cpp server, text-generation-webui all speak this same protocol)
works through it via the `base_url` override. That's what makes "works
for both API and locally-hosted models" a one-parameter difference
rather than two separate client implementations.

Default model: gpt-5-nano — OpenAI's cheapest text-capable model as of
2026-09 ($0.05 / $0.40 per 1M input/output tokens), appropriate for the
classification/summarization-shaped tasks Layer 4 performs. Note GPT-5-
family models reject the `temperature` parameter and use
`max_completion_tokens` instead of the legacy `max_tokens` — handled
automatically here based on model name, not left for callers to hit.

Swap backends via environment variables (see get_default_client()):
    LLM_PROVIDER=openai   (default)
    LLM_MODEL=gpt-5-nano  (default)
    OPENAI_API_KEY=...    (required for the real API)

    # Point at a local OpenAI-compatible server instead, e.g. Ollama:
    LLM_MODEL=llama3
    LLM_BASE_URL=http://localhost:11434/v1
    OPENAI_API_KEY=not-needed

A second, optional model can be configured for reasoning-heavy tasks
(see get_reasoning_client()) — e.g. LLM_REASONING_MODEL=gpt-5-mini for
stronger reasoning on Layer 4's root-cause analysis, while the cheaper
default model handles simpler classification-shaped tasks.

These are read from a `.env` file at the project root if one exists (see
`.env.example`) — loaded automatically on import via python-dotenv, so
no manual `export` is needed. Falls back silently to real environment
variables (e.g. set by a deployment platform) if python-dotenv isn't
installed or no `.env` file is found.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient(ABC):
    """
    Minimal provider-agnostic interface for a chat-capable LLM backend.
    Concrete implementations wrap a specific provider/protocol. Callers
    depend only on this interface.
    """

    model: str = "unknown"

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Run one chat completion turn.

        Parameters
        ----------
        system : str
            System prompt (ground rules, task instructions).
        messages : list of {"role": "user"|"assistant", "content": str}
            Conversation so far, ending with the newest user turn.
        json_mode : bool
            Ask the backend to constrain output to a JSON object, where
            supported. Callers must still parse defensively — not every
            backend enforces this, and even ones that do can occasionally
            emit malformed JSON.
        temperature, max_tokens : optional generation parameters. A
            backend that doesn't support one silently ignores it rather
            than erroring, so callers don't need to know per-model quirks.

        Returns
        -------
        str
            The assistant's reply text.
        """
        raise NotImplementedError

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        One chat turn where the model may reply with text OR request a
        tool call (OpenAI function-calling wire format for `tools`).

        Not abstract: a backend that doesn't support tool-calling (or a
        test double that doesn't need to exercise this path) simply
        doesn't override it. Callers should catch `NotImplementedError`
        and fall back to plain `complete()`.

        Returns
        -------
        dict
            {"role": "text", "content": str} if the model just replied,
            or {"role": "tool_call", "tool_name": str, "arguments": dict,
            "content": Optional[str]} if it requested a tool call. Only
            the first tool call is returned — callers here never need a
            model to request more than one in a single turn.
        """
        raise NotImplementedError


# GPT-5 / o-series "reasoning" models reject `temperature` and require
# `max_completion_tokens` instead of the legacy Chat Completions
# `max_tokens` parameter. Detected from the model name so the common
# case (gpt-5-nano) works correctly without the caller needing to know.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAICompatibleClient(LLMClient):
    """
    Talks to any backend implementing OpenAI's Chat Completions API —
    the real OpenAI API, or a self-hosted OpenAI-compatible server via
    `base_url`.
    """

    DEFAULT_MODEL = "gpt-5-nano"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        supports_temperature: Optional[bool] = None,
        max_tokens_param: Optional[str] = None,
        **default_kwargs: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleClient. "
                "Install with: pip install openai"
            )

        self.model = model
        is_reasoning_model = model.startswith(_REASONING_MODEL_PREFIXES)
        self.supports_temperature = (
            not is_reasoning_model if supports_temperature is None else supports_temperature
        )
        self.max_tokens_param = max_tokens_param or (
            "max_completion_tokens" if is_reasoning_model else "max_tokens"
        )
        self._default_kwargs = default_kwargs

        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or "not-needed"
        )
        resolved_base_url = base_url or os.environ.get("LLM_BASE_URL")
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = dict(self._default_kwargs)
        if temperature is not None and self.supports_temperature:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs[self.max_tokens_param] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self._default_kwargs)
        if temperature is not None and self.supports_temperature:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs[self.max_tokens_param] = max_tokens

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=tools,
            **kwargs,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return {
                "role": "tool_call",
                "tool_name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "content": msg.content,
            }
        return {"role": "text", "content": msg.content or ""}


class MockLLMClient(LLMClient):
    """
    Canned-response client for tests/offline development — no network
    calls, no API key needed. Records every call for assertions and
    returns responses from a supplied queue (or "{}"/"" if exhausted).
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        tool_call_responses: Optional[List[Dict[str, Any]]] = None,
    ):
        self.model = "mock"
        self._responses = list(responses) if responses else []
        self._tool_call_responses = list(tool_call_responses) if tool_call_responses else []
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.calls.append({
            "system": system, "messages": messages, "json_mode": json_mode,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        if self._responses:
            return self._responses.pop(0)
        return "{}" if json_mode else ""

    def complete_with_tools(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.calls.append({
            "system": system, "messages": messages, "tools": tools,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        if self._tool_call_responses:
            return self._tool_call_responses.pop(0)
        if self._responses:
            return {"role": "text", "content": self._responses.pop(0)}
        return {"role": "text", "content": ""}


def get_default_client() -> LLMClient:
    """
    Build an LLMClient from environment configuration. This is the
    intended way to obtain a client — code written against it never
    needs to change to switch providers.

        LLM_PROVIDER   "openai" (default) or "mock"
        LLM_MODEL      model name (default: gpt-5-nano)
        LLM_BASE_URL   override for a local/self-hosted OpenAI-compatible
                       server (e.g. http://localhost:11434/v1 for Ollama)
        OPENAI_API_KEY required for the real OpenAI API; local servers
                       usually accept any placeholder value
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return MockLLMClient()
    model = os.environ.get("LLM_MODEL", OpenAICompatibleClient.DEFAULT_MODEL)
    base_url = os.environ.get("LLM_BASE_URL")
    return OpenAICompatibleClient(model=model, base_url=base_url)


def get_reasoning_client() -> LLMClient:
    """
    Build the LLMClient for reasoning-heavy tasks — currently Layer 4's
    root_cause_analysis, which must weigh conflicting evidence (temporal
    onset order vs. causal-graph direction vs. drift magnitude) rather
    than do straightforward extraction/classification like the other
    Layer 4 methods.

    Reads LLM_REASONING_MODEL; falls back to the same model as
    get_default_client() (LLM_MODEL, or gpt-5-nano) if unset, so nothing
    changes unless you opt in. Same LLM_PROVIDER/LLM_BASE_URL as
    get_default_client() — only the model differs.

    Example: set LLM_REASONING_MODEL=gpt-5-mini (or gpt-5) in .env for a
    stronger model on root-cause analysis while keeping cheaper
    gpt-5-nano for column identity / variable type / cluster role
    inference.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return MockLLMClient()
    model = (
        os.environ.get("LLM_REASONING_MODEL")
        or os.environ.get("LLM_MODEL", OpenAICompatibleClient.DEFAULT_MODEL)
    )
    base_url = os.environ.get("LLM_BASE_URL")
    return OpenAICompatibleClient(model=model, base_url=base_url)
