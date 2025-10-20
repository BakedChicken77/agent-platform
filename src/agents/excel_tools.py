# tools/excel_tools.py
from __future__ import annotations
import io
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
except Exception:
    pd = None

from langchain_core.tools import tool

# =========================
# Helpers & Data IO
# =========================
def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)

def _require_pandas() -> Optional[str]:
    if pd is None:
        return _err("pandas (and openpyxl/pyarrow for Excel/Parquet) is required.")
    return None

def _to_records_json(df: "pd.DataFrame") -> str:
    return json.dumps({"data": df.to_dict(orient="records")}, ensure_ascii=False)

def _from_records_json(data_json: str) -> "pd.DataFrame":
    obj = json.loads(data_json or "{}")
    rows = obj.get("data")
    if rows is None:
        raise ValueError("Expected JSON with key 'data' (list of row dicts).")
    return pd.DataFrame(rows)

# --- NEW: cross-platform path normalization (Windows & Linux) ---
def _norm_path(path: str) -> str:
    """
    Normalize incoming paths so both Windows (e.g., C:\\dir\\file.xlsx, UNC)
    and Linux (/home/user/file.csv) forms work transparently.
    - Strips wrapping quotes
    - Expands ~
    - Converts file:// URIs
    - Normalizes separators
    """
    if not isinstance(path, str):
        return path
    p = path.strip().strip('"').strip("'")
    if p.lower().startswith("file://"):
        p = p[7:]
    p = os.path.expanduser(p)
    return os.path.normpath(p)
# ----------------------------------------------------------------

def _to_markdown_table(df: "pd.DataFrame", max_rows: int = 50) -> str:
    if df.empty:
        return "| (empty) |\n|---|\n| *(no rows)* |"
    trimmed = df.head(max_rows)
    # Escape pipes/newlines
    safe = trimmed.applymap(
        lambda x: str(x).replace("|", r"\|").replace("\n", " ") if not pd.isna(x) else ""
    )
    header = "| " + " | ".join(map(str, safe.columns)) + " |\n"
    sep = "| " + " | ".join(["---"] * safe.shape[1]) + " |\n"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for _, row in safe.iterrows())
    note = "" if len(df) <= max_rows else f"\n\n*…{len(df)-max_rows} more rows omitted…*"
    return header + sep + body + note

def _read_excel_like(
    path: str,
    sheet: Union[str, int, None],
    header_row: Union[int, str, None],
    skip_rows: Union[int, List[int], None],
    use_cols: Union[str, List[Union[int, str]], None],
    cell_range: Optional[str],
    na_values: Optional[List[str]],
    dtype: Optional[Dict[str, str]],
    date_cols: Optional[List[str]],
    thousands: Optional[str],
    decimal: Optional[str],
) -> "pd.DataFrame":
    # Normalize path for Windows/Linux
    path = _norm_path(path)

    # Handle CSV transparently; for XLS/XLSX use pandas read_excel
    ext = os.path.splitext(path)[1].lower()
    if cell_range and ext not in (".xlsx", ".xlsm", ".xlsb", ".xls"):
        raise ValueError("cell_range is only supported for Excel files, not CSV.")
    if header_row == "infer":
        header = 0
    elif header_row is None:
        header = None
    else:
        header = int(header_row)

    # Convert "A1:H200" to usecols and skiprows/ nrows if header is above the range
    kw: Dict[str, Any] = {}
    if na_values:
        kw["na_values"] = na_values
    if dtype:
        kw["dtype"] = dtype
    if thousands:
        kw["thousands"] = thousands
    if decimal:
        kw["decimal"] = decimal
    if date_cols:
        kw["parse_dates"] = date_cols
        kw["infer_datetime_format"] = True

    if ext in (".csv", ".txt"):
        if use_cols:
            kw["usecols"] = use_cols
        if skip_rows is not None:
            kw["skiprows"] = skip_rows
        df = pd.read_csv(path, header=header, **kw)
    else:
        # Excel
        if use_cols:
            kw["usecols"] = use_cols
        if skip_rows is not None:
            kw["skiprows"] = skip_rows
        if cell_range:
            kw["usecols"], kw["skiprows"], kw["nrows"] = _range_to_reader_args(cell_range, header)
        df = pd.read_excel(path, sheet_name=sheet, header=header, **kw)

    # If header=None, synthesize column names
    if df.columns.dtype == "int64" or any(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    return df

def _col_to_index(col: str) -> int:
    # Convert Excel column letters (e.g., "AB") to 0-based index
    col = col.upper()
    n = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid column letter: {col}")
        n = n * 26 + (ord(ch) - 64)
    return n - 1

def _parse_a1(a1: str) -> Tuple[int, int]:
    # returns (row_idx0, col_idx0) 0-based
    m = re.fullmatch(r"\s*([A-Za-z]+)(\d+)\s*", a1)
    if not m:
        raise ValueError(f"Invalid A1 cell: {a1}")
    col, row = m.group(1), int(m.group(2))
    return row - 1, _col_to_index(col)

def _range_to_reader_args(a1_range: str, header: Optional[int]) -> Tuple[str, List[int], int]:
    # Convert "B3:H200" into (usecols, skiprows, nrows) for pandas readers
    m = re.fullmatch(r"\s*([A-Za-z]+\d+):([A-Za-z]+\d+)\s*", a1_range)
    if not m:
        raise ValueError(f"Invalid A1 range: {a1_range}")
    r1, c1 = _parse_a1(m.group(1))
    r2, c2 = _parse_a1(m.group(2))
    if r2 < r1 or c2 < c1:
        raise ValueError("Invalid A1 range: end before start.")
    # usecols as letters range, skiprows as list(range(0, r1)) unless header is within
    # Compute letters from c1..c2
    def idx_to_letters(idx: int) -> str:
        s = ""
        idx += 1
        while idx:
            idx, rem = divmod(idx - 1, 26)
            s = chr(65 + rem) + s
        return s
    usecols = f"{idx_to_letters(c1)}:{idx_to_letters(c2)}"
    # If header is None or >= r1, we can skip rows before r1
    if header is None or (isinstance(header, int) and header >= r1):
        skiprows = list(range(0, r1))
        nrows = r2 - r1 + 1
    else:
        # Header lies above r1; don't drop it, just limit nrows
        skiprows = None
        nrows = r2 + 1  # read through end row
    return usecols, skiprows, nrows

def _maybe_strip_whitespace(df: "pd.DataFrame", columns: Optional[List[str]], do_strip: bool) -> "pd.DataFrame":
    if not do_strip:
        return df
    cols = columns if columns else df.columns.tolist()
    for c in cols:
        if c in df.columns and pd.api.types.is_object_dtype(df[c].dtype):
            df[c] = df[c].astype(str).str.strip()
    return df

def _coerce_types(df: "pd.DataFrame", types: Dict[str, str]) -> Tuple["pd.DataFrame", List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    for col, tname in (types or {}).items():
        if col not in df.columns:
            issues.append({"column": col, "row": None, "error": "missing_column_for_coercion"})
            continue
        try:
            if tname in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif tname in ("float", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif tname in ("str", "string"):
                df[col] = df[col].astype("string").fillna("")
            elif tname in ("bool", "boolean"):
                df[col] = df[col].astype("boolean")
            elif tname in ("datetime", "date"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                issues.append({"column": col, "row": None, "error": f"unsupported_target_type:{tname}"})
        except Exception as e:
            issues.append({"column": col, "row": None, "error": f"type_coercion_failed:{e}"})
    return df, issues

# =========================
# Tools
# =========================

@tool
def EXCEL_FindHeaders(config_json: str) -> str:
    """
    Scan the top of a sheet to guess the header row.
    Input (JSON):
      {
        "path": "<file.xlsx|csv>",
        "sheet": "Sheet1" | 0 | null,
        "search_rows": 50,               # how many rows to scan
        "min_distinct": 3,               # min non-empty distinct cells to consider header-ish
        "prefer_text": true              # prefer rows with mostly text-like cells
      }
    Output JSON: {"header_row": <int|null>, "score": <float>, "preview": [[...],[...],...]}
    """
    err = _require_pandas()
    if err: return err
    try:
        cfg = json.loads(config_json or "{}")
        path = _norm_path(cfg["path"])  # normalize for Windows/Linux
        sheet = cfg.get("sheet", 0)
        search_rows = int(cfg.get("search_rows", 50))
        min_distinct = int(cfg.get("min_distinct", 3))
        prefer_text = bool(cfg.get("prefer_text", True))

        # Read small top window without header
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt"):
            sample = pd.read_csv(path, nrows=search_rows, header=None)
        else:
            sample = pd.read_excel(path, sheet_name=sheet, nrows=search_rows, header=None)

        best = (-1.0, None)  # (score, row_idx)
        for r in range(min(search_rows, len(sample))):
            row = sample.iloc[r]
            vals = [str(x).strip() for x in row if pd.notna(x) and str(x).strip() != ""]
            distinct = len(set(vals))
            if distinct < min_distinct:
                continue
            textish = sum(1 for v in vals if re.search(r"[A-Za-z]", v))
            frac_text = textish / max(1, len(vals))
            score = distinct + (0.5 * frac_text if prefer_text else 0.0)
            if score > best[0]:
                best = (score, r)

        out = {
            "header_row": best[1],
            "score": best[0],
            "preview": sample.head(min(10, search_rows)).astype(str).fillna("").values.tolist()
        }
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        return _err(f"Header detection failed: {e}")

@tool
def EXCEL_Read(config_json: str) -> str:
    """
    Read Excel/CSV into JSON records with robust options.
    Input (JSON):
      {
        "path": "<file.xlsx|csv>",
        "sheet": "Sheet1"|0|null,
        "header_row": 0|"infer"|null,
        "skip_rows": 0|[0,1,...],
        "use_cols": "A:D"|["A","B","C"]|[0,2,5],
        "cell_range": "B3:H200",         # Excel only
        "na_values": ["", "NA", "-"],
        "dtype": {"Amount":"float64","Qty":"int"},
        "date_cols": ["Date"],
        "thousands": ",",
        "decimal": ".",
        "strip_whitespace": true
      }
    Output JSON: {"data":[{...}, ...]}
    """
    err = _require_pandas()
    if err: return err
    try:
        cfg = json.loads(config_json or "{}")
        df = _read_excel_like(
            path=_norm_path(cfg["path"]),
            sheet=cfg.get("sheet", 0),
            header_row=cfg.get("header_row", "infer"),
            skip_rows=cfg.get("skip_rows"),
            use_cols=cfg.get("use_cols"),
            cell_range=cfg.get("cell_range"),
            na_values=cfg.get("na_values"),
            dtype=cfg.get("dtype"),
            date_cols=cfg.get("date_cols"),
            thousands=cfg.get("thousands"),
            decimal=cfg.get("decimal"),
        )
        df = _maybe_strip_whitespace(df, None, bool(cfg.get("strip_whitespace", True)))
        return _to_records_json(df)
    except Exception as e:
        return _err(f"Read failed: {e}")

@tool
def EXCEL_MapSchema(payload_json: str, mapping_json: str) -> str:
    """
    Normalize/rename/coerce/filter rows/cols according to per-workflow mapping rules.
    payload_json: {"data":[...]}  (from EXCEL_Read or prior step)
    mapping_json example:
      {
        "rename": {"Invoice #":"invoice_id","Amt":"amount"},
        "required": {"invoice_id":"", "amount":0.0},          # add if missing with default
        "select": ["invoice_id","date","amount","notes"],     # final column order (optional)
        "strip_columns": ["invoice_id","notes"],              # strip whitespace on these columns
        "coerce_types": {"amount":"float64","date":"datetime","invoice_id":"string"},
        "drop_rows_where_null_all": ["invoice_id","amount"],  # drop rows where all listed are null/empty
        "drop_duplicates_on": ["invoice_id"],                 # keep first
        "replace": { "notes": { "N/A": "" } },                # exact value replacements per column
        "regex_replace": { "invoice_id": [ ["^#",""], ["\\s+",""] ] }  # list of [pattern,repl]
      }
    Output JSON: {"data":[...], "issues":[...]}  (issues = type coercion or mapping warnings)
    """
    err = _require_pandas()
    if err: return err
    try:
        df = _from_records_json(payload_json)
        m = json.loads(mapping_json or "{}")

        # rename
        if m.get("rename"):
            df = df.rename(columns=m["rename"])

        # add required defaults
        for col, default in (m.get("required") or {}).items():
            if col not in df.columns:
                df[col] = default

        # simple value replacements
        repl = m.get("replace") or {}
        for col, mapping in repl.items():
            if col in df.columns and isinstance(mapping, dict):
                df[col] = df[col].replace(mapping)

        # regex replacements
        rx = m.get("regex_replace") or {}
        for col, rules in rx.items():
            if col in df.columns and isinstance(rules, list):
                s = df[col].astype(str)
                for pat, repl_val in rules:
                    s = s.str.replace(pat, repl_val, regex=True)
                df[col] = s

        # drop rows where specific columns are all null/blank
        if m.get("drop_rows_where_null_all"):
            cols = [c for c in m["drop_rows_where_null_all"] if c in df.columns]
            mask_all_blank = df[cols].applymap(lambda x: (pd.isna(x)) or (str(x).strip() == "")).all(axis=1)
            df = df[~mask_all_blank].copy()

        # strip whitespace on selected columns
        df = _maybe_strip_whitespace(df, m.get("strip_columns"), True)

        # coerce types
        issues: List[Dict[str, Any]] = []
        if m.get("coerce_types"):
            df, t_issues = _coerce_types(df, m["coerce_types"])
            issues.extend(t_issues)

        # drop duplicates
        if m.get("drop_duplicates_on"):
            keys = [c for c in m["drop_duplicates_on"] if c in df.columns]
            if keys:
                df = df.drop_duplicates(subset=keys, keep="first")

        # final column selection/order
        if m.get("select"):
            cols = [c for c in m["select"] if c in df.columns]
            df = df.reindex(columns=cols)

        return json.dumps({"data": df.to_dict(orient="records"), "issues": issues}, ensure_ascii=False)
    except Exception as e:
        return _err(f"MapSchema failed: {e}")

@tool
def EXCEL_Validate(payload_json: str, rules_json: str) -> str:
    """
    Validate rows against simple declarative rules.
    rules_json example:
      {
        "columns": {
          "invoice_id": {"not_null": true, "regex": "^[A-Za-z0-9\\-]+$"},
          "amount": {"not_null": true, "min": 0.0, "max": 100000.0},
          "date": {"not_null": true}   # if datetime coercion failed earlier, you'll see NaT here
        },
        "unique": ["invoice_id"]   # cross-row uniqueness
      }
    Output JSON: {"ok": true|false, "violations": [ {row:int, column:str|null, rule:str, value:any} ]}
    """
    err = _require_pandas()
    if err: return err
    try:
        df = _from_records_json(payload_json)
        rules = json.loads(rules_json or "{}")
        col_rules = rules.get("columns", {})
        uniq_cols = rules.get("unique", []) or []

        violations: List[Dict[str, Any]] = []

        # per-column checks
        for col, cr in col_rules.items():
            if col not in df.columns:
                violations.append({"row": None, "column": col, "rule": "missing_column", "value": None})
                continue
            s = df[col]
            if cr.get("not_null"):
                bad = s.isna() | (s.astype(str).str.strip() == "")
                for idx in df.index[bad]:
                    violations.append({"row": int(idx), "column": col, "rule": "not_null", "value": None})
            if "min" in cr:
                try:
                    bad = pd.to_numeric(s, errors="coerce") < float(cr["min"])
                    for idx, val in df.loc[bad, col].items():
                        violations.append({"row": int(idx), "column": col, "rule": "min", "value": val})
                except Exception:
                    pass
            if "max" in cr:
                try:
                    bad = pd.to_numeric(s, errors="coerce") > float(cr["max"])
                    for idx, val in df.loc[bad, col].items():
                        violations.append({"row": int(idx), "column": col, "rule": "max", "value": val})
                except Exception:
                    pass
            if "regex" in cr:
                pat = re.compile(cr["regex"])
                bad = ~s.astype(str).fillna("").apply(lambda x: bool(pat.search(x)))
                for idx, val in df.loc[bad, col].items():
                    violations.append({"row": int(idx), "column": col, "rule": "regex", "value": val})

        # uniqueness across rows
        for col in uniq_cols:
            if col in df.columns:
                dup_mask = df.duplicated(subset=[col], keep="first")
                for idx, val in df.loc[dup_mask, col].items():
                    violations.append({"row": int(idx), "column": col, "rule": "unique", "value": val})
            else:
                violations.append({"row": None, "column": col, "rule": "missing_column_for_unique", "value": None})

        return json.dumps({"ok": len(violations) == 0, "violations": violations}, ensure_ascii=False)
    except Exception as e:
        return _err(f"Validate failed: {e}")

@tool
def EXCEL_Write(payload_json: str, output_json: str) -> str:
    """
    Write JSON records to Excel/CSV.
    payload_json: {"data":[...]} (from previous step)
    output_json example:
      {
        "path": "out.xlsx" | "out.csv",
        "sheet": "Normalized",                 # for Excel
        "if_exists": "replace_file"|"new_file",# current impl: replace_file only
        "index": false,
        "engine": "openpyxl"                   # for xlsx
      }
    Output JSON: {"wrote":"<path>", "rows": <int>}
    """
    err = _require_pandas()
    if err: return err
    try:
        df = _from_records_json(payload_json)
        cfg = json.loads(output_json or "{}")
        path = _norm_path(cfg["path"])  # normalize for Windows/Linux
        index = bool(cfg.get("index", False))
        ext = os.path.splitext(path)[1].lower()

        if ext in (".csv", ".txt"):
            df.to_csv(path, index=index)
        else:
            sheet = cfg.get("sheet", "Sheet1")
            engine = cfg.get("engine", None)
            with pd.ExcelWriter(path, engine=engine) as xlw:
                df.to_excel(xlw, sheet_name=sheet, index=index)
        return json.dumps({"wrote": path, "rows": int(len(df))}, ensure_ascii=False)
    except Exception as e:
        return _err(f"Write failed: {e}")

@tool
def EXCEL_ToMarkdown(payload_json: str, max_rows: int = 50) -> str:
    """
    Render JSON records as a Markdown table (for user-facing summaries).
    Returns a string (markdown table).
    """
    err = _require_pandas()
    if err: return err
    try:
        df = _from_records_json(payload_json)
        return _to_markdown_table(df, max_rows=max_rows)
    except Exception as e:
        return _err(f"ToMarkdown failed: {e}")

@tool
def EXCEL_ToCSVBytes(payload_json: str) -> str:
    """
    Return CSV as base64 string for transport between tools (if needed).
    Output: {"base64":"..."}
    """
    import base64
    err = _require_pandas()
    if err: return err
    try:
        df = _from_records_json(payload_json)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return json.dumps({"base64": base64.b64encode(buf.getvalue()).decode("ascii")}, ensure_ascii=False)
    except Exception as e:
        return _err(f"ToCSVBytes failed: {e}")

# Export convenient list for ToolNode([...])
tools = [
    EXCEL_FindHeaders,
    EXCEL_Read,
    EXCEL_MapSchema,
    EXCEL_Validate,
    EXCEL_Write,
    EXCEL_ToMarkdown,
    EXCEL_ToCSVBytes,
]
