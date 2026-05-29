from typing import TypedDict


class AgentState(TypedDict):
    customer_query: str
    query_category: str
    query_sentiment: str
    escalation_customer_info: dict
    on_call_support_info: dict
    final_response: str
