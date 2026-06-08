"""LLM provider configuration for the contact center agent."""

from __future__ import annotations

import os
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> Union[ChatOpenAI, ChatAnthropic]:
    """Get an LLM instance based on provider and model configuration.

    Args:
        provider: LLM provider name ("openai" or "anthropic").
            Falls back to ``LLM_PROVIDER`` env var, then ``openai``.
        model: Model identifier. Falls back to ``LLM_MODEL`` env var,
            then a provider-specific default.
        temperature: Sampling temperature (default 0.3).

    Returns:
        Configured LangChain chat model instance.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        default_model = "gpt-4o-mini"
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", default_model),
            temperature=temperature,
        )

    if provider in ("anthropic", "claude"):
        default_model = "claude-3-5-sonnet-20241022"
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", default_model),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
