from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.state import AgentState
from app.models import QueryCategory, QuerySentiment
from app.llm import get_llm
from app.vectorstore import load_vectorstore

# Load shared resources once at module import time.
# This means the embeddings model and FAISS index are loaded a single time
# when the app starts, not on every query.
_llm        = get_llm()
_vectorstore = load_vectorstore()


# ---------------------------------------------------------------------------
# Node 1 — Classify the query into a healthcare department
# ---------------------------------------------------------------------------

def categorize_inquiry(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template("""
You are a customer support agent for a healthcare company.

Classify the customer query into exactly one of these departments:
- billing: payment, charges, invoices, bills, refunds, payment methods
- appointments: booking, rescheduling, cancelling, doctor availability
- records: accessing, sharing, updating, downloading medical records
- insurance: policy, coverage, claims, plan updates, insurance questions

Return only the best matching category.

Customer query: {customer_query}
""")
    structured = _llm.with_structured_output(QueryCategory)
    result = structured.invoke(prompt.format_messages(customer_query=state["customer_query"]))
    state["query_category"] = result.category
    return state


# ---------------------------------------------------------------------------
# Node 2 — Detect the emotional tone of the query
# ---------------------------------------------------------------------------

def analyze_sentiment(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template("""
You are a healthcare customer support agent.

Classify the sentiment of the customer query into exactly one of:
- distress: health emergency or urgent medical danger
- negative: unhappy, frustrated, angry, dissatisfied
- neutral: normal question without strong emotion
- positive: satisfaction, appreciation, clearly positive tone

Return only the best matching sentiment.

Customer query: {customer_query}
""")
    structured = _llm.with_structured_output(QuerySentiment)
    result = structured.invoke(prompt.format_messages(customer_query=state["customer_query"]))
    state["query_sentiment"] = result.sentiment
    return state


# ---------------------------------------------------------------------------
# Node 3 — RAG: retrieve department docs and generate a support answer
# ---------------------------------------------------------------------------

def generate_department_response(state: AgentState) -> AgentState:
    category = state["query_category"]
    query    = state["customer_query"]

    if category not in ["billing", "appointments", "records", "insurance"]:
        state["final_response"] = (
            "Apologies, I was not able to answer your question. "
            "Please reach out to our support team."
        )
        return state

    # ⚠️ Updated from deprecated Chroma metadata filter dict
    # → FAISS does not support dict-style filter= in similarity_search.
    # Instead: fetch more candidates, then filter by metadata manually.
    candidates = _vectorstore.similarity_search(query, k=10)
    docs = [d for d in candidates if d.metadata.get("category") == category][:3]

    if not docs:
        state["final_response"] = (
            "Apologies, I was not able to find relevant information for your query. "
            "Please reach out to our support team."
        )
        return state

    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful healthcare support agent for the {department} department.

Use the knowledge base information below to answer the customer query clearly and helpfully.
If the information does not help, say:
"Apologies, I was not able to answer your question. Please reach out to our support team."

Customer query:
{customer_query}

Knowledge base:
{context}
""")

    chain = prompt | _llm | StrOutputParser()
    state["final_response"] = chain.invoke(
        {"department": category, "customer_query": query, "context": context}
    )
    return state


# ---------------------------------------------------------------------------
# Node 4 — Collect name, phone, email for human-agent escalation
# ---------------------------------------------------------------------------

def accept_escalation_input(state: AgentState) -> AgentState:
    print("\n--- Escalation Form ---")
    state["escalation_customer_info"] = {
        "name":   input("Your name: ").strip(),
        "number": input("Phone number: ").strip(),
        "email":  input("Email address: ").strip(),
    }
    return state


# ---------------------------------------------------------------------------
# Node 5 — Build the human-escalation response
# ---------------------------------------------------------------------------

def escalate_to_human(state: AgentState) -> AgentState:
    info = state["escalation_customer_info"]
    name   = info.get("name", "Customer")
    email  = info.get("email", "your email")
    number = info.get("number", "your phone number")
    state["final_response"] = (
        f"Apologies, {name}. We are sorry for the inconvenience. "
        f"Someone from our support team will reach out to you at {email}. "
        f"If needed, they will also call you at {number}."
    )
    return state


# ---------------------------------------------------------------------------
# Node 6 — Collect name and phone for on-call emergency
# ---------------------------------------------------------------------------

def accept_on_call_input(state: AgentState) -> AgentState:
    print("\n--- Emergency On-Call Form ---")
    state["on_call_support_info"] = {
        "name":   input("Your name: ").strip(),
        "number": input("Phone number: ").strip(),
    }
    return state


# ---------------------------------------------------------------------------
# Node 7 — Build the on-call emergency response
# ---------------------------------------------------------------------------

def escalate_to_on_call(state: AgentState) -> AgentState:
    info   = state["on_call_support_info"]
    name   = info.get("name", "Patient")
    number = info.get("number", "your phone number")
    state["final_response"] = (
        f"Don't worry, {name}. "
        f"Someone from our on-call team of expert doctors will reach out to you "
        f"shortly at {number} for immediate assistance."
    )
    return state
