# LangGraph Healthcare Router

A LangGraph-based query routing system for a fictional healthcare support desk. It classifies incoming customer queries by **department** and **sentiment**, then routes each query to one of three branches: an automated RAG answer, a human-agent escalation, or an emergency on-call response.

Built as a learning project while studying LangGraph's conditional routing and state management. The dataset is a synthetic healthcare FAQ covering billing, appointments, insurance, and medical records.

---

## How It Works

```
Customer query
    |
    v
[categorize_inquiry]        -- LLM classifies: billing / appointments / records / insurance
    |
    v
[analyze_sentiment]         -- LLM classifies: positive / neutral / negative / distress
    |
    +--- negative  --> [accept_escalation_input] --> [escalate_to_human]   --> END
    |
    +--- distress  --> [accept_on_call_input]    --> [escalate_to_on_call] --> END
    |
    +--- positive/neutral --> [generate_department_response]               --> END
                                      |
                            FAISS retrieval (filtered by department)
                                      |
                              LLM generates answer
```

### Key Implementation Decisions

- **FAISS over Chroma** — FAISS does not support passing a `filter={}` dict directly to `similarity_search`. This project fetches the top-10 candidates and post-filters by `metadata["category"]` in Python. This is reliable and dependency-free.
- **Structured output for classification** — Both the category and sentiment nodes use `llm.with_structured_output(PydanticModel)`, which forces the LLM to return only the allowed enum values and prevents hallucinated labels.
- **Vectorstore built separately** — `build_vectorstore.py` is a one-time setup script. The app loads the saved index at startup, which is much faster than rebuilding it on every run.
- **Department name normalisation** — The original FAQ JSON uses `"medical_records"` as a department key, but the LLM classifier returns `"records"`. These are normalised to `"records"` at index-build time.

---

## Project Structure

```
langgraph-healthcare-router/
├── main.py                  # interactive CLI entry point
├── build_vectorstore.py     # one-time setup: builds and saves the FAISS index
├── healthcare_faq.json      # 20 synthetic Q&A pairs across 4 departments
├── requirements.txt
├── .env.example
├── .gitignore
└── app/
    ├── __init__.py
    ├── state.py             # AgentState TypedDict
    ├── models.py            # QueryCategory and QuerySentiment Pydantic models
    ├── llm.py               # ChatGroq client factory
    ├── vectorstore.py       # FAISS loader with clear error if index is missing
    ├── nodes.py             # all 7 node functions
    └── graph.py             # StateGraph wiring and compile
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/imbhaskarm/langgraph-healthcare-router.git
cd langgraph-healthcare-router
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> First install downloads the `sentence-transformers/all-MiniLM-L6-v2` model (~90 MB). This only happens once.

**3. Set up your Groq API key**

Copy `.env.example` to `.env` and add your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [https://console.groq.com](https://console.groq.com)

**4. Build the FAISS vector index (run once)**

```bash
python build_vectorstore.py
```

This reads `healthcare_faq.json`, embeds all 20 documents, and saves the index to `healthcare_faiss_db/`. You only need to do this once (or after changing the FAQ data).

**5. Run the app**

```bash
python main.py
```

---

## Example Queries

**Normal billing question → RAG answer:**
```
You: What payment methods do you accept?
Category : billing
Sentiment: neutral

Response:
We accept all major credit cards (Visa, MasterCard, American Express), debit cards,
checks, cash, online payments through our patient portal, as well as HSA and FSA cards.
```

**Frustrated patient → human escalation:**
```
You: I have been waiting 3 weeks for a callback and no one has helped me.
Category : appointments
Sentiment: negative

--- Escalation Form ---
Your name: Jane
Phone number: 9876543210
Email address: jane@example.com

Response:
Apologies, Jane. We are sorry for the inconvenience. Someone from our support team
will reach out to you at jane@example.com. If needed, they will also call you at 9876543210.
```

**Medical distress → on-call team:**
```
You: I am having severe chest pain and can't breathe.
Category : appointments
Sentiment: distress

--- Emergency On-Call Form ---
Your name: John
Phone number: 9876543210

Response:
Don't worry, John. Someone from our on-call team of expert doctors will reach out to
you shortly at 9876543210 for immediate assistance.
```

---

## Things I Learned Building This

- LangGraph routes via `add_conditional_edges` by returning a node name string from a plain Python function. The routing map dict is just for validation — the string returned must match a key in it.
- FAISS and Chroma have different filter APIs. Chroma accepts `where={"category": "billing"}` natively; FAISS does not support dict-style metadata filters in `similarity_search`. The correct approach for FAISS is to over-fetch and filter in Python.
- `with_structured_output(PydanticModel)` is far more reliable than asking the LLM to return JSON and parsing it yourself. Groq's Llama models handle Pydantic constraints well when the schema is simple.
- Loading the vectorstore and LLM once at module level (rather than inside each node function) makes a big practical difference — no repeated model downloads or connection overhead per query.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangGraph | StateGraph, conditional routing, node wiring |
| Groq (Llama 3.3-70b) | Classification (category + sentiment) and RAG answer generation |
| FAISS | Local vector similarity search |
| HuggingFace sentence-transformers | Free embedding model (`all-MiniLM-L6-v2`) |
| python-dotenv | Environment variable management |

---

## GitHub Repository Description

> LangGraph healthcare query router — classifies customer queries by department and sentiment, then routes to RAG answers, human escalation, or emergency on-call response.

**Topics:** `langgraph` `langchain` `rag` `faiss` `groq`
