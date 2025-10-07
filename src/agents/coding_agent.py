# coding_agent.py

import os
from dotenv import load_dotenv
from core import get_model, settings
from langgraph.prebuilt import create_react_agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
import math
import numpy as np
import plotly
import io, base64
import matplotlib
matplotlib.use("Agg")                 # force headless backend
import matplotlib.pyplot as plt
import plotly.io as pio

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
model = get_model(settings.DEFAULT_MODEL)

# safe_globals = {
#     "__builtins__": {"print": print},
#     "math": math,
#     "numpy": np,
#     "np": np,  # <-- optional, for convenience
#     "plotly": plotly,
# }
safe_globals = {
    "__builtins__": {"print": print},
    "math": math,
    "numpy": np,
    "np": np,
    "plotly": plotly,
    "pio": pio,
    "plt": plt,
    "io": io,
    "base64": base64,
}


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return its stdout or an error message.
    """
    repl = PythonREPL(globals=safe_globals)#, _locals={})
    try:
        return repl.run(code)
    except Exception as e:
        return f"Execution failed: {e}"



# prompt_coding_agent = """\
# You are a senior Python coding assistant specialized in data-science and exploratory data analysis.
# Follow these rules:
# - Generate correct, runnable Python code.
# - Always wrap your code in a single python_repl tool call.
# - If the user asks for explanation, design advice, or non-code discussion, respond directly in plain text.
# - The runtime already includes: `math`, `numpy` as `np` and `plotly`. Do NOT use `import` statements — \
# they are unnecessary and disabled. Simply write code assuming those modules are already available.
# - DO NOT attempt to display plots via popup window. Instead, create .png's for all plots and save them in the workspace.
# """

prompt_coding_agent = """\
You are a senior Python coding assistant specialized in data-science and plotting.

Rules:
- Wrap ALL runnable code in a single python_repl tool call.
- Prefer Plotly. If you create a Plotly figure `fig`, finish by:
    print("PLOTLY_JSON:" + fig.to_json())
- If you use Matplotlib (via preloaded `plt`), finish by writing the figure to an in-memory PNG:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    import base64  # already available
    print("DATA_URI:data:image/png;base64," + base64.b64encode(buf.read()).decode())
- Do NOT call plt.show(). Do NOT open GUI windows. Do NOT write files to disk.
- Only print one payload line (PLOTLY_JSON:... or DATA_URI:...).
"""

 
# Create the coding expert agent
coding_agent = create_react_agent(
    model=model,
    tools=[python_repl],
    name="coding_expert",
    prompt=prompt_coding_agent
).with_config(tags=["skip_stream"])
