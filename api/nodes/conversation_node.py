import sys
import os

# Add llm-proxy to path for importing our shared router
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../llm-proxy')))

from agent_router import get_dev_router
from escalation_manager import ConversationEscalationManager

# Initialize the router and manager once for the LangGraph application
router = get_dev_router()
escalation_manager = ConversationEscalationManager(router)

def conversation_node(state: dict):
    """
    LangGraph Node that handles the primary conversation flow.
    It reads the emotional state and conversation history from the LangGraph state,
    and uses the EscalationManager to dynamically route the LLM request.
    """
    messages = state.get("messages", [])
    
    # In a real LangGraph flow, these would be populated by an earlier
    # SentimentAnalysis node or fetched from Redis
    emotion_state = state.get("emotion_state", {
        "frustration": 0.0,
        "anger": 0.0,
        "complexity": 0.0
    })
    objection_count = state.get("objection_count", 0)

    # Use the escalation manager to get the response
    response = escalation_manager.generate_response(
        messages=messages,
        emotion_state=emotion_state,
        objection_count=objection_count
    )

    if response["status"] == "error":
        # Handle failure (e.g. transfer to human)
        return {"messages": [{"role": "assistant", "content": "I apologize, I need to transfer you to a human supervisor."}]}

    # Return the updated state
    return {
        "messages": [{"role": "assistant", "content": response["content"]}],
        "model_used": response["model_used"]
    }
