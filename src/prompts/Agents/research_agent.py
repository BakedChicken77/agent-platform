# construct the research agent prompt using the index context, tool lists, and tool usage instructions

from pydantic import BaseModel
from schema import ProgramContext


def RESEARCH_AGENT_PROMPT(index_context:ProgramContext, tools:BaseModel, tool_usage_instructions=None) -> str:

    # create string describing tools and descriptions
    tool_info = []
    for tool in tools:
        tool_info.append(f"""* **`{tool.name}`** — {tool.description}""")
    
    tool_description = "\n".join(tool_info)

    # added 'Tool Usage Instructions' header to string if tool_usage_instructions is provided
    if tool_usage_instructions:
        final_tool_usage_instructions = f"""
### Tool Usage Instructions:
{tool_usage_instructions}
"""
    else:
        final_tool_usage_instructions=""

    #construct prompt
    return f"""\
You are a highly capable research assistant for Leonardo DRS Inc. in Fort Walton Beach, Florida, known for its advanced \
defense technology solutions. You have extensive experience in Electrical Engineering, Program Management, Systems Engineering, and Project Engineering. \
You are tasked with researching and answering questions about Leonardo DRS' {index_context.full_program_name} Documentation.
---

You have access to the following tool(s):
{tool_description}
{final_tool_usage_instructions}
NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
---

{index_context.detailed_overview}
---

Use the tools available to you to research the question and generate a comprehensive and informative answer for the given question based solely on the search results. 

You have access to the following tool(s):
{tool_description}
{final_tool_usage_instructions}
NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
---

### Response Guidelines:
* Cite your sources with **markdown-formatted links** using **only URLs returned by the tools**. Limit to **1-2 citations** per response unless more are essential.
* Be direct, thorough, and precise. **Do not speculate.** Only respond based on verified content from retrieved documents.
"""










if __name__ == "__main__":
    from prompts.indexes.jacske import JACSKE
    from langchain_core.tools import BaseTool, tool


    tool_usage_instructions = """\
1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `JACSKE_Get_Full_Doc_Text`.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.\
"""

    def fake_tool1(query: str) -> str:
        """fake tool 1"""
        return f"fake_tool_1 returned after getting query: {query}"
    def fake_tool2(query: str) -> str:
        """fake tool 2"""
        return f"fake_tool_2 returned after getting query: {query}"

    fake_tool_1: BaseTool = tool(fake_tool1)
    fake_tool_1.name = "FAKE_TOOL_1_NAME"  
    fake_tool_1.description = "FAKE_TOOL_1_DESCRIPTION"  

    fake_tool_2: BaseTool = tool(fake_tool2)
    fake_tool_2.name = "FAKE_TOOL_2_NAME"  
    fake_tool_2.description = "FAKE_TOOL_2_DESCRIPTION"  

    tools = [fake_tool_1, fake_tool_2]
    print(RESEARCH_AGENT_PROMPT(JACSKE, tools,tool_usage_instructions))
