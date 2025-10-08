import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agents import coding_agent
from schema.files import FileMeta


@pytest.fixture
def sample_excel_file(tmp_path: Path) -> tuple[Path, bytes]:
    df = pd.DataFrame({"value": [1, 2, 3]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    data = buffer.getvalue()
    file_path = tmp_path / "uploaded-file"
    file_path.write_bytes(data)
    return file_path, data


def test_python_repl_exposes_uploaded_excel(monkeypatch: Any, sample_excel_file: tuple[Path, bytes]) -> None:
    file_path, data = sample_excel_file

    meta = FileMeta(
        id="file-123",
        user_id="user-1",
        thread_id="thread-1",
        tenant_id=None,
        original_name="numbers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        size=len(data),
        sha256="stub",
        path=str(file_path),
        created_at=0,
        indexed=False,
    )

    def fake_list_metadata(user_id: str, thread_id: str | None) -> list[FileMeta]:
        assert user_id == "user-1"
        assert thread_id == "thread-1"
        return [meta]

    monkeypatch.setattr(coding_agent.catalog_postgres, "list_metadata", fake_list_metadata)

    state = {"configurable": {"user_id": "user-1", "thread_id": "thread-1"}}

    result_paths = coding_agent.python_repl.func(
        "print(sorted(uploaded_file_paths.keys()))",
        state,
    )
    assert "numbers.xlsx" in result_paths
    assert "file-123" in result_paths

    result_metadata = coding_agent.python_repl.func(
        "print(list_uploaded_files()[0]['path'])",
        state,
    )
    assert str(file_path) in result_metadata

    analysis_output = coding_agent.python_repl.func(
        "\n".join(
            [
                "buffer = load_uploaded_file('numbers.xlsx')",
                "df = pd.read_excel(buffer)",
                "print(df.to_dict(orient='records'))",
            ]
        ),
        state,
    )
    assert "{'value': 1}" in analysis_output
    assert "{'value': 3}" in analysis_output
