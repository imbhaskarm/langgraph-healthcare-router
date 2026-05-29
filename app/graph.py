from langgraph.graph import StateGraph, START, END

from app.state import AgentState
from app.nodes import (
    categorize_inquiry,
    analyze_sentiment,
    generate_department_response,
    accept_escalation_input,
    escalate_to_human,
    accept_on_call_input,
    escalate_to_on_call,
)


def _route(state: AgentState) -> str:
    """
    Conditional edge: decides which node to run after sentiment analysis.

    negative  → collect escalation details (human agent)
    distress  → collect emergency contact (on-call doctor)
    positive / neutral → answer via RAG
    """
    sentiment = state["query_sentiment"]
    if sentiment == "negative":
        return "accept_escalation_input"
    elif sentiment == "distress":
        return "accept_on_call_input"
    else:
        return "generate_department_response"


def build_graph():
    """Assemble and compile the LangGraph workflow."""
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("categorize_inquiry",         categorize_inquiry)
    g.add_node("analyze_sentiment",           analyze_sentiment)
    g.add_node("generate_department_response", generate_department_response)
    g.add_node("accept_escalation_input",     accept_escalation_input)
    g.add_node("escalate_to_human",           escalate_to_human)
    g.add_node("accept_on_call_input",        accept_on_call_input)
    g.add_node("escalate_to_on_call",         escalate_to_on_call)

    # Fixed edges
    g.add_edge(START, "categorize_inquiry")
    g.add_edge("categorize_inquiry", "analyze_sentiment")

    # Conditional routing after sentiment analysis
    g.add_conditional_edges(
        "analyze_sentiment",
        _route,
        {
            "generate_department_response": "generate_department_response",
            "accept_escalation_input":      "accept_escalation_input",
            "accept_on_call_input":         "accept_on_call_input",
        },
    )

    # Branch completions
    g.add_edge("generate_department_response", END)
    g.add_edge("accept_escalation_input", "escalate_to_human")
    g.add_edge("escalate_to_human", END)
    g.add_edge("accept_on_call_input", "escalate_to_on_call")
    g.add_edge("escalate_to_on_call", END)

    return g.compile()
