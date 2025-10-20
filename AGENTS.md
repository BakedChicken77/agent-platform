# Agent Instructions

Use `uv run` or first activate `.venv` with `source .venv/bin/activate`

To run frontend:
```sh
uv run streamlit run src/streamlit_app.py
```

To run backend:
```sh
uv run fastapi dev src/service/service.py
```

Always run the regression suite before submitting a change:
```bash
uv run pytest
```
