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
    """Get an LLM instance based on provider and model configuration."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    
    # If the user has configured the LiteLLM proxy base URL, use it
    # and pass the call_id dynamically for LangSmith/Langfuse tagging via the proxy
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if base_url:
        from api.tracing import current_call_id
        call_id = current_call_id.get() or "unknown_call"
        
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "conv-primary"),
            temperature=temperature,
            api_key="sk-placeholder", # LiteLLM proxy doesn't need real key
            base_url=base_url,
            default_headers={"x-call-id": call_id} # Pass call_id to proxy
        )

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
