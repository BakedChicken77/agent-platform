# universal_supervisor_agent.py

import os
from dotenv import load_dotenv
from core import get_model, settings
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor, create_handoff_tool
from agents.tools import calculator, JACSKE_database_search, JACSKE_get_full_doc_text
from agents.coding_agent import coding_agent  # import the coding expert
from langchain_experimental.tools.python.tool import PythonREPLTool
from prompts.Agents.research_agent import RESEARCH_AGENT_PROMPT
from prompts.indexes.jacske import JACSKE

# Load environment variables
load_dotenv()

# Initialize LLM
model = get_model(settings.DEFAULT_MODEL)

# Simple math tools
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Math expert
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])


researcher_tools = [JACSKE_database_search, JACSKE_get_full_doc_text]
research_tool_usage_instructions = """\
1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `JACSKE_Get_Full_Doc_Text`.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.\
"""
researcher_prompt = RESEARCH_AGENT_PROMPT(
    index_context = JACSKE, 
    tools = researcher_tools,
    tool_usage_instructions = research_tool_usage_instructions
)

# Research expert
research_agent = create_react_agent(
    model=model,
    tools=[JACSKE_database_search, JACSKE_get_full_doc_text],
    name="research_expert",
    prompt=researcher_prompt,
).with_config(tags=["skip_stream"])



# Supervisor workflow with all three experts
workflow = create_supervisor(
    [research_agent, math_agent, coding_agent],
    tools=[
        create_handoff_tool(agent_name="math_expert", name="transfer_to_math_expert", description="Transfer task to math expert", add_handoff_messages = True),
        create_handoff_tool(agent_name="research_expert", name="transfer_to_research_expert", description="Transfer task to research expert", add_handoff_messages = True),
        create_handoff_tool(agent_name="coding_expert", name="transfer_to_coding_expert", description="Transfer task to coding expert", add_handoff_messages = True)
    ],
    # add_handoff_messages = False,
    model=model,
    prompt="""\
You are a supervisor for defense contractor, known for its advanced \
defense technology solutions. You are tasked with managing three experts to assist technicians \
with questions about program documentation.

The three experts you are managing are:
 • research_expert - documentation research  
 • math_expert     - arithmetic & calculations  
 • coding_expert   - writing and debugging Python code, data exploration, plotting  

Delegate each user request to the single most suitable expert, return their response, 
and only intervene to summarize or if none of the experts applies.

NOTE: THE USER CAN'T SEE THE EXPERTS' RESPONSES.\
""",
    add_handoff_back_messages=False,
)

universal_supervisor_agent = workflow.compile()
universal_supervisor_agent.name = "JACSKE_Agent_Coding"
