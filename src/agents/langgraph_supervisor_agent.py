
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from agents.tools import calculator, database_search, get_full_doc_text

from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

# research_agent = create_react_agent(
#     model=model,
#     tools=[web_search],
#     name="research_expert",
#     prompt="You are a world class researcher with access to web search. Do not do any math.",
# ).with_config(tags=["skip_stream"])


research_agent = create_react_agent(
    model=model,
    tools=[database_search, get_full_doc_text],
    name="research_expert",
    prompt=f"""\
You are a highly capable research assistant. You have access to the following tools:

* **`Database_Search`** — Performs semantic search across Leonardo DRS, Inc.'s official Employee Handbook and Operational Process documents.
* **`Get_Full_Doc_Text`** — Retrieves the complete text of a document by its exact filename (e.g., `'SEP-04-01(M) Process for Product Development.docx'`).

---

### Tool Usage Instructions:

1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `Get_Full_Doc_Text`.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
---

### Response Guidelines:

* Cite your sources with **markdown-formatted links** using **only URLs returned by the tools**. Limit to **1-2 citations** per response unless more are essential.
* Be direct, thorough, and precise. **Do not speculate.** Only respond based on verified content from retrieved documents.
""",
).with_config(tags=["skip_stream"])


# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        f"""\
You are a team supervisor managing a research expert and a math expert to support employees by researching \
Leonardo DRS, Inc official Employee Handbook and Operational Process documents.
For Employee Handbook and Operational Process information, use research_agent.
For math problems, use math_agent.\
"""
    ),
    add_handoff_back_messages=False,
)




langgraph_supervisor_agent = workflow.compile()

