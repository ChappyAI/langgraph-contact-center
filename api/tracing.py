import os
from contextvars import ContextVar

# A ContextVar to keep track of the current call ID for LangSmith tracing
current_call_id = ContextVar("current_call_id", default=None)

def get_tracing_metadata():
    """
    Returns a dictionary to be used as `metadata` in LangGraph invocations.
    This injects the contact center call_id natively into LangSmith.
    """
    call_id = current_call_id.get()
    return {"call_id": call_id} if call_id else {}
