"""LLM provider configuration for the contact center agent."""

from __future__ import annotations

import os
from typing import Any, Union

from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import PermissionDeniedError as AnthropicPermissionDeniedError
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import PermissionDeniedError as OpenAIPermissionDeniedError

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
_TERMINAL_PROVIDER_EXCEPTIONS = (
    AnthropicAuthenticationError,
    AnthropicPermissionDeniedError,
    OpenAIAuthenticationError,
    OpenAIPermissionDeniedError,
)


def _provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    selected_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    if model:
        selected_model = model
    elif provider is None:
        selected_model = os.getenv("LLM_MODEL", "")
    else:
        selected_model = ""
    if not selected_model:
        selected_model = (
            _DEFAULT_ANTHROPIC_MODEL
            if selected_provider in ("anthropic", "claude")
            else _DEFAULT_OPENAI_MODEL
        )
    return selected_provider, selected_model


def resolve_llm_config(
    provider: str | None = None,
    model: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> tuple[str, str, str, str]:
    """Resolve primary and fallback provider settings."""
    selected_provider, selected_model = _provider_model(provider, model)
    fallback_provider, fallback_model = _provider_model(
        fallback_provider or os.getenv("LLM_FALLBACK_PROVIDER", "anthropic"),
        fallback_model or os.getenv("LLM_FALLBACK_MODEL"),
    )
    return selected_provider, selected_model, fallback_provider, fallback_model


def is_terminal_provider_error(exc: BaseException) -> bool:
    """Return whether a provider error should skip retries."""
    if isinstance(exc, _TERMINAL_PROVIDER_EXCEPTIONS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429,) or (isinstance(status, int) and status >= 500):
        return False
    if status in (401, 403):
        return True

    name = type(exc).__name__.lower()
    if any(
        marker in name for marker in ("authentication", "permission", "unauthorized")
    ):
        return True

    message = str(exc).lower()
    if any(
        marker in message
        for marker in ("rate limit", "rate_limit", "too many requests")
    ):
        return False
    if any(marker in message for marker in ("http 401", "http 403")):
        return True
    return any(
        marker in message
        for marker in (
            "no credits remaining",
            "insufficient_quota",
            "insufficient quota",
            "quota exceeded",
            "credit balance",
            "invalid_api_key",
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "permission denied",
            "forbidden",
        )
    )


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> Union[ChatOpenAI, ChatAnthropic]:
    """Get an LLM instance based on provider and model configuration."""
    selected_provider, selected_model = _provider_model(provider, model)

    if selected_provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "temperature": temperature,
        }
        if base_url:
            from api.tracing import current_call_id

            kwargs.update(
                {
                    "api_key": "sk-placeholder",
                    "base_url": base_url,
                    "default_headers": {
                        "x-call-id": current_call_id.get() or "unknown_call"
                    },
                }
            )
        return ChatOpenAI(**kwargs)

    if selected_provider in ("anthropic", "claude"):
        return ChatAnthropic(model=selected_model, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {selected_provider}")


def get_llm_chain(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> list[Union[ChatOpenAI, ChatAnthropic]]:
    """Build the requested LLM followed by its configured fallback."""
    primary, primary_model, fallback, fallback_model = resolve_llm_config(
        provider, model, fallback_provider, fallback_model
    )
    chain = [get_llm(primary, primary_model, temperature)]
    if (primary, primary_model) != (fallback, fallback_model):
        chain.append(get_llm(fallback, fallback_model, temperature))
    return chain
