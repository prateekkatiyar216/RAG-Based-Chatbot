"""
Excel structured-data analysis.

Excel files are NOT pushed through the text-chunking / FAISS RAG pipeline.
Instead, each sheet is kept as a pandas DataFrame in memory, and numeric /
tabular questions are answered by running controlled pandas operations
(never by an LLM inventing numbers, and never via eval()/exec() on
LLM-generated code). The LLM's job is limited to:
  1. classifying whether a question needs the document RAG path, the Excel
     path, or both, and
  2. turning a natural-language question into a small structured JSON
     "plan" that this module then executes with pandas.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
MAX_TABLE_ROWS = 50          # cap on rows sent to the LLM / shown in a table
PREVIEW_ROWS = 8             # rows shown in the "Excel preview" panel


# ─────────────────────────────────────────────────────────────────────────
# LLM access (separate from RAGSearch so Excel Q&A doesn't require a PDF
# knowledge base to have been built first)
# ─────────────────────────────────────────────────────────────────────────
def get_llm(model_name: str = DEFAULT_LLM_MODEL) -> ChatGroq:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Add it to your .env file:\n  GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────
class ExcelLoadError(Exception):
    pass


def load_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load every sheet of a workbook into a dict of DataFrames.

    Empty / unreadable sheets are dropped rather than crashing the app.
    Raises ExcelLoadError with a friendly message if the workbook itself
    can't be read at all.
    """
    try:
        raw_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        raise ExcelLoadError(f"couldn't read this Excel file — it may be corrupted. ({e})")

    if not raw_sheets:
        raise ExcelLoadError("the workbook appears to be empty (no sheets found).")

    clean_sheets = {}
    for name, df in raw_sheets.items():
        if df is None or df.empty or df.shape[1] == 0:
            continue
        # Drop fully-empty rows/columns that Excel sometimes leaves behind.
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        clean_sheets[str(name)] = df

    if not clean_sheets:
        raise ExcelLoadError("none of the sheets in this workbook contain usable data.")

    return clean_sheets


# ─────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────
def get_file_metadata(filename: str, sheets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    sheet_meta = []
    for name, df in sheets.items():
        sheet_meta.append({
            "name": name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        })
    return {"filename": filename, "sheets": sheet_meta}


def build_sheet_catalog_text(excel_data: Dict[str, Dict[str, pd.DataFrame]]) -> str:
    """A compact text description of every file/sheet/column, used as LLM context."""
    lines = []
    for filename, sheets in excel_data.items():
        lines.append(f"File: {filename}")
        for sheet_name, df in sheets.items():
            cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
            lines.append(f"  Sheet: {sheet_name} — {df.shape[0]} rows × {df.shape[1]} cols")
            lines.append(f"    Columns: {cols}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Helpers: column / sheet resolution (fuzzy, case-insensitive — never
# invents a column that doesn't exist)
# ─────────────────────────────────────────────────────────────────────────
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    norm_target = _normalize(name)
    for c in df.columns:
        if _normalize(c) == norm_target:
            return c
    # partial match fallback
    for c in df.columns:
        if norm_target in _normalize(c) or _normalize(c) in norm_target:
            return c
    return None


def resolve_sheet(
    excel_data: Dict[str, Dict[str, pd.DataFrame]],
    file_hint: Optional[str],
    sheet_hint: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[pd.DataFrame]]:
    """Pick the best matching (filename, sheet_name, dataframe)."""
    candidates: List[Tuple[str, str, pd.DataFrame]] = []
    for filename, sheets in excel_data.items():
        for sheet_name, df in sheets.items():
            candidates.append((filename, sheet_name, df))

    if not candidates:
        return None, None, None

    if file_hint:
        norm_file = _normalize(file_hint)
        candidates = [c for c in candidates if norm_file in _normalize(c[0])] or candidates

    if sheet_hint:
        norm_sheet = _normalize(sheet_hint)
        sheet_matches = [c for c in candidates if norm_sheet in _normalize(c[1])]
        if sheet_matches:
            candidates = sheet_matches

    if len(candidates) == 1:
        return candidates[0]

    # Still ambiguous — fall back to the largest sheet, which is usually
    # the "main" data sheet the user means.
    candidates.sort(key=lambda c: c[2].shape[0], reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────
# Question routing (PDF vs Excel vs mixed) — only invoked when BOTH a
# document knowledge base and Excel data are loaded.
# ─────────────────────────────────────────────────────────────────────────
_ROUTE_SYSTEM_PROMPT = """You route a user's question to the right data source.
Respond with ONLY a JSON object: {"route": "pdf" | "excel" | "mixed"}
- "pdf": the question is about narrative/document content (papers, reports, explanations).
- "excel": the question needs calculations, filtering, sorting, or lookups over spreadsheet data (sums, averages, totals, top N, comparisons between rows/columns, "which X has the most Y", etc).
- "mixed": the question explicitly asks to combine/compare information from the document AND the spreadsheet.
No prose, no markdown fences — just the JSON object."""


def classify_question(question: str, llm: Optional[ChatGroq] = None) -> str:
    """Returns 'pdf', 'excel', or 'mixed'. Defaults to 'mixed' on any failure
    so both sources get consulted rather than silently dropping one."""
    try:
        llm = llm or get_llm()
        resp = llm.invoke([
            SystemMessage(content=_ROUTE_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
        parsed = _extract_json(resp.content)
        route = str(parsed.get("route", "")).lower()
        if route in ("pdf", "excel", "mixed"):
            return route
    except Exception:
        pass
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────
# Question → structured plan (LLM classifies/interprets; pandas computes)
# ─────────────────────────────────────────────────────────────────────────
_PLAN_SYSTEM_TEMPLATE = """You turn a user's question about spreadsheet data into a JSON "plan".
You do NOT calculate anything yourself — you only identify which operation and columns to use.

Available data:
{catalog}

Respond with ONLY a JSON object, no prose, no markdown fences, matching this shape:
{{
  "file": "<filename or null>",
  "sheet": "<sheet name or null>",
  "operation": "sum" | "mean" | "count" | "min" | "max" | "median" | "groupby" | "filter" | "sort_top" | "sort_bottom" | "compare" | "describe" | "list_sheets" | "list_columns" | "clarify",
  "agg": "sum" | "mean" | "count" | "min" | "max" | "median",
  "value_column": "<column name or null>",
  "group_column": "<column name or null>",
  "group_column_2": "<a second column name, or null — only for a two-dimension breakdown like 'per line per day'>",
  "filter_column": "<column name or null>",
  "filter_op": ">" | "<" | ">=" | "<=" | "==" | "!=" | null,
  "filter_value": "<value or null>",
  "sort_column": "<column name or null>",
  "top_n": <integer or null>,
  "compare_column": "<column name to group by for a comparison, or null>",
  "compare_values": ["<value1>", "<value2>"] or null,
  "clarification_question": "<a short question to ask the user, only if operation is clarify>"
}}

Rules:
- Only use column/sheet names that actually appear in the data above. Never invent one.
- If the question is ambiguous between two clearly-different columns (e.g. Target vs Actual) and it matters, set operation to "clarify" and ask a short clarification question.
- "Which X has the most/least Y" -> operation "groupby" with group_column=X, value_column=Y, agg="sum", sort implied descending (top_n=1) unless asked for more.
- "which X is most common", "most frequent X", "highest number of <categorical field>", "top X reasons/types" -> operation "groupby" with BOTH group_column and value_column set to that same categorical field X — this is interpreted as a count of how often each value occurs.
- "average/mean Y per X" -> operation "groupby", agg="mean", group_column=X, value_column=Y.
- "Y per X per Z" or "Y broken down by X and Z" or "Y for each X, separately by Z" -> operation "groupby" with group_column=X and group_column_2=Z (a two-dimension breakdown). Only set group_column_2 when the question clearly asks for two dimensions at once.
- Default "agg" to "sum" unless the question asks for an average/mean, a count, a min, a max, or a median.
- "top N records" -> operation "sort_top" with top_n.
- "show/filter records where ..." -> operation "filter".
- "compare A and B" -> operation "compare".
- "what sheets are there" -> operation "list_sheets".
- "what columns are in X" -> operation "list_columns".
- Keep every field short. Never explain your reasoning anywhere in the response — the JSON object is the entire response, nothing before or after it.
"""


class _PlanParseError(Exception):
    """Raised when the model's plan response can't be turned into JSON."""


def _extract_json(text: str) -> dict:
    """Pull the first complete, balanced JSON object out of an LLM response.

    Deliberately NOT a greedy regex: LLMs routinely add a trailing sentence
    after the JSON (despite being told not to), and that sentence can
    itself contain stray '{' or '}' characters (e.g. "...broken down by
    {day} instead"). A greedy "first { to last }" match would swallow that
    trailing text into the JSON and corrupt it. Instead we scan char by
    char, tracking string/escape state so braces inside quoted strings
    don't affect nesting depth, and stop at the first object that balances.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    start = text.find("{")
    if start == -1:
        raise _PlanParseError("The response didn't contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end is None:
        raise _PlanParseError("The response's JSON object looked incomplete.")

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise _PlanParseError(f"Couldn't parse the response as JSON ({e}).") from e


def plan_question(question: str, excel_data: Dict[str, Dict[str, pd.DataFrame]], llm: Optional[ChatGroq] = None) -> dict:
    catalog = build_sheet_catalog_text(excel_data)
    llm = llm or get_llm()
    resp = llm.invoke([
        SystemMessage(content=_PLAN_SYSTEM_TEMPLATE.format(catalog=catalog)),
        HumanMessage(content=question),
    ])
    return _extract_json(resp.content)




# ─────────────────────────────────────────────────────────────────────────
# Deterministic markdown table rendering (no LLM involved — guarantees the
# numbers shown match what pandas actually computed)
# ─────────────────────────────────────────────────────────────────────────
def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS) -> str:
    if df.empty:
        return "_No matching rows._"
    truncated = df.shape[0] > max_rows
    view = df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        cells = [str(v).replace("|", "/") for v in row.tolist()]
        rows.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, sep] + rows)
    if truncated:
        table += f"\n\n_Showing {max_rows} of {df.shape[0]} rows._"
    return table


# ─────────────────────────────────────────────────────────────────────────
# Plan execution — the ONLY place calculations actually happen
# ─────────────────────────────────────────────────────────────────────────
def _coerce_filter_value(series: pd.Series, value: Any) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


_FILTER_OPS = {
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}

_VALID_AGGS = {"sum", "mean", "count", "min", "max", "median"}


def execute_plan(plan: dict, excel_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Runs the plan with pandas. Returns a dict describing the result:
    {"kind": "clarify"|"error"|"scalar"|"table"|"text", ...}
    Never raises — all failures come back as a friendly "error" result.
    """
    operation = plan.get("operation")

    if operation == "clarify":
        return {"kind": "clarify", "message": plan.get("clarification_question") or "Could you clarify which data you mean?"}

    filename, sheet_name, df = resolve_sheet(excel_data, plan.get("file"), plan.get("sheet"))
    if df is None:
        return {"kind": "error", "message": "I don't have any Excel data loaded to answer that."}

    try:
        if operation == "list_sheets":
            names = []
            for fname, sheets in excel_data.items():
                for sname, sdf in sheets.items():
                    names.append(f"**{sname}** ({fname}) — {sdf.shape[0]} rows × {sdf.shape[1]} cols")
            return {"kind": "text", "message": "\n".join(names)}

        if operation == "list_columns":
            return {"kind": "text", "message": f"Columns in **{sheet_name}**: " + ", ".join(df.columns)}

        if operation in ("sum", "mean", "count", "min", "max", "median"):
            col = resolve_column(df, plan.get("value_column"))
            if operation != "count" and col is None:
                return {"kind": "error", "message": f"I couldn't find a column matching '{plan.get('value_column')}' in **{sheet_name}**."}
            if operation == "count":
                result = int(df.shape[0]) if col is None else int(df[col].count())
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                result = getattr(series, operation)()
                if pd.isna(result):
                    return {"kind": "error", "message": f"'{col}' doesn't look like a numeric column, so I can't compute {operation}."}
            return {"kind": "scalar", "value": result, "column": col, "sheet": sheet_name, "operation": operation}

        if operation == "groupby":
            group_col = resolve_column(df, plan.get("group_column"))
            value_col = resolve_column(df, plan.get("value_column"))
            group_col_2 = resolve_column(df, plan.get("group_column_2")) if plan.get("group_column_2") else None
            if group_col_2 == group_col:
                group_col_2 = None
            if not group_col or not value_col:
                return {"kind": "error", "message": "I couldn't find the columns needed for that grouping."}

            top_n = plan.get("top_n") or 10
            agg = plan.get("agg") or "sum"
            if agg not in _VALID_AGGS:
                agg = "sum"

            group_cols = [group_col] + ([group_col_2] if group_col_2 else [])
            is_numeric = pd.api.types.is_numeric_dtype(df[value_col])

            if group_col == value_col and not group_col_2:
                # "which X is most common" -> frequency of each category.
                counts = df[group_col].value_counts().head(top_n)
                result_df = pd.DataFrame({group_col: counts.index, "Count": counts.values})

            elif agg == "count" or not is_numeric:
                computed = df.groupby(group_cols)[value_col].count()
                result_value_name = "Count"
                if len(group_cols) == 1:
                    computed = computed.sort_values(ascending=False).head(top_n)
                    result_df = pd.DataFrame({group_col: computed.index, result_value_name: computed.values})
                else:
                    idx = computed.index
                    data = {gc: idx.get_level_values(i) for i, gc in enumerate(group_cols)}
                    data[result_value_name] = computed.values
                    result_df = pd.DataFrame(data).sort_values(group_cols).head(MAX_TABLE_ROWS)

            else:
                series = pd.to_numeric(df[value_col], errors="coerce")
                computed = df.assign(**{value_col: series}).groupby(group_cols)[value_col].agg(agg)
                result_value_name = value_col if value_col not in group_cols else f"{value_col} ({agg})"
                if len(group_cols) == 1:
                    computed = computed.sort_values(ascending=False).head(top_n)
                    result_df = pd.DataFrame({group_col: computed.index, result_value_name: computed.values})
                else:
                    idx = computed.index
                    data = {gc: idx.get_level_values(i) for i, gc in enumerate(group_cols)}
                    data[result_value_name] = computed.values
                    result_df = pd.DataFrame(data).sort_values(group_cols).head(MAX_TABLE_ROWS)

            return {"kind": "table", "dataframe": result_df, "sheet": sheet_name}

        if operation == "filter":
            col = resolve_column(df, plan.get("filter_column"))
            op = plan.get("filter_op")
            value = plan.get("filter_value")
            if not col or op not in _FILTER_OPS or value is None:
                return {"kind": "error", "message": "I couldn't work out how to filter that — could you rephrase with a specific column and value?"}
            series = df[col]
            coerced = _coerce_filter_value(series, value)
            compare_series = pd.to_numeric(series, errors="coerce") if isinstance(coerced, float) else series
            mask = _FILTER_OPS[op](compare_series, coerced)
            result_df = df[mask.fillna(False)]
            return {"kind": "table", "dataframe": result_df, "sheet": sheet_name}

        if operation in ("sort_top", "sort_bottom"):
            col = resolve_column(df, plan.get("sort_column") or plan.get("value_column"))
            if not col:
                return {"kind": "error", "message": "I couldn't find a column to sort by."}
            ascending = operation == "sort_bottom"
            series = pd.to_numeric(df[col], errors="coerce")
            top_n = plan.get("top_n") or 10
            result_df = df.assign(**{f"_sort_{col}": series}).sort_values(f"_sort_{col}", ascending=ascending).drop(columns=[f"_sort_{col}"]).head(top_n)
            return {"kind": "table", "dataframe": result_df, "sheet": sheet_name}

        if operation == "compare":
            group_col = resolve_column(df, plan.get("compare_column"))
            value_col = resolve_column(df, plan.get("value_column"))
            if not group_col or not value_col:
                return {"kind": "error", "message": "I couldn't find the columns needed for that comparison."}

            if group_col != value_col and pd.api.types.is_numeric_dtype(df[value_col]):
                series = pd.to_numeric(df[value_col], errors="coerce")
                agg = df.assign(**{value_col: series}).groupby(group_col)[value_col].agg(["sum", "mean", "count"])
                result_df = pd.DataFrame({
                    group_col: agg.index,
                    "sum": agg["sum"].values,
                    "mean": agg["mean"].values,
                    "count": agg["count"].values,
                })
            else:
                counts = df.groupby(group_col)[value_col].count()
                result_df = pd.DataFrame({group_col: counts.index, "count": counts.values})

            values = plan.get("compare_values")
            if values:
                norm_values = [_normalize(v) for v in values]
                result_df = result_df[result_df[group_col].apply(lambda x: _normalize(x) in norm_values)]
            return {"kind": "table", "dataframe": result_df, "sheet": sheet_name}

        if operation == "describe":
            return {"kind": "table", "dataframe": df.describe(include="all").reset_index(), "sheet": sheet_name}

    except Exception as e:
        return {"kind": "error", "message": f"I ran into a problem analyzing that data ({e})."}

    return {"kind": "error", "message": "I couldn't work out how to answer that from the spreadsheet data."}


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────
def analyze_excel_question(question: str, excel_data: Dict[str, Dict[str, pd.DataFrame]], llm: Optional[ChatGroq] = None) -> str:
    if not excel_data:
        return "No Excel data has been loaded yet."

    llm = llm or get_llm()

    try:
        plan = plan_question(question, excel_data, llm=llm)
    except _PlanParseError:
        # The model's JSON came back malformed or truncated — this is
        # usually a one-off glitch, so retry once before giving up.
        try:
            plan = plan_question(question, excel_data, llm=llm)
        except _PlanParseError:
            return "⚠️ I had trouble understanding that question against the spreadsheet data — could you try rephrasing it more simply?"
    except EnvironmentError:
        raise
    except Exception as e:
        return f"⚠️ I had trouble understanding that question against the spreadsheet data ({e})."

    result = execute_plan(plan, excel_data)

    if result["kind"] == "clarify":
        return result["message"]
    if result["kind"] == "error":
        return f"⚠️ {result['message']}"
    if result["kind"] == "text":
        return result["message"]

    if result["kind"] == "scalar":
        value = result["value"]
        display_value = f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
        summary_context = f"The {result['operation']} of '{result['column']}' in sheet '{result['sheet']}' is {display_value}."
        try:
            explanation = llm.invoke([
                SystemMessage(content="Explain this computed result to the user in one short, natural sentence. Do not change or recompute the number."),
                HumanMessage(content=f"Question: {question}\nComputed result: {summary_context}"),
            ]).content
            return explanation
        except Exception:
            return summary_context

    if result["kind"] == "table":
        table_md = dataframe_to_markdown(result["dataframe"])
        try:
            intro = llm.invoke([
                SystemMessage(content="Write one short, natural sentence introducing the table below that answers the user's question. Do not list numbers yourself — the table already shows them."),
                HumanMessage(content=f"Question: {question}\nSheet used: {result['sheet']}"),
            ]).content.strip()
        except Exception:
            intro = f"Here's what I found in **{result['sheet']}**:"
        return f"{intro}\n\n{table_md}"

    return "⚠️ I couldn't work out how to answer that from the spreadsheet data."
