# streamlit_app.py
import os
from datetime import datetime
from typing import List, Dict, Any
import re
import json

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser

# Local import: must be in the same folder with this exact name
# Your scraper/pipeline module (the "working" version you’re using)
import event_pipeline as ep


# =========================
# Page + Layout
# =========================
st.set_page_config(page_title="InnovatePGH Event Scraper", layout="wide")
st.title("InnovatePGH – Event Scraper + Chat")

# =========================
# OpenAI (from secrets, with optional override)
# =========================
def init_openai_from_secrets():
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    # Optional model overrides via secrets
    if "OPENAI_MODEL_EXTRACT" in st.secrets:
        os.environ["OPENAI_MODEL_EXTRACT"] = st.secrets["OPENAI_MODEL_EXTRACT"]
    if "OPENAI_MODEL_SUMMARY" in st.secrets:
        os.environ["OPENAI_MODEL_SUMMARY"] = st.secrets["OPENAI_MODEL_SUMMARY"]

    return api_key

def allow_manual_override(api_key):
    # Collapsible section for dev override
    with st.expander("OpenAI Settings", expanded=False):
        manual_key = st.text_input(
            "OpenAI API Key (overrides secrets for this session)",
            type="password",
            help="Optional – only needed if you want to override the key in secrets."
        )
        if manual_key:
            api_key = manual_key
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("OpenAI API key loaded for this session.")
    return api_key

api_key = init_openai_from_secrets()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("OpenAI API key loaded from secrets. LLM extraction + summaries ENABLED.")
    st.caption(
        f"Models: extract={os.environ.get('OPENAI_MODEL_EXTRACT','gpt-4o-mini')}, "
        f"summary={os.environ.get('OPENAI_MODEL_SUMMARY','gpt-4o-mini')}"
    )
else:
    st.warning("No API key found in Streamlit secrets. Fallback heuristics will be used (summaries generic).")

api_key = allow_manual_override(api_key)

# Session storage for results across tabs
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=ep.FINAL_COLS)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict(role, content)]


# =========================
# Small utilities
# =========================
def _clean_cell(val) -> str:
    # Convert NaN/None to "", and "nan" (string) to ""
    if isinstance(val, str):
        s = val.strip()
        return "" if s.lower() == "nan" else s
    if pd.isna(val):
        return ""
    return str(val).strip()

def load_uploaded_df(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def asof_parse(s: str) -> datetime:
    return dateparser.parse(s)

def show_quick_metrics(df: pd.DataFrame):
    with_date = (df["Date"].astype(str).str.len() > 0).sum() if not df.empty else 0
    with_url = (df["Event URL"].astype(str).str.len() > 0).sum() if not df.empty else 0
    st.caption(f"Rows: {len(df)} | With Date: {with_date} | With URL: {with_url}")


# =========================
# Tab: Scrape
# =========================
tab_scrape, tab_chat = st.tabs(["🕷️ Scrape", "💬 Chat with Results"])

with tab_scrape:
    st.subheader("Scrape event listings")
    st.markdown("Upload a CSV/XLSX with **Org name** and **URL** columns.")

    uploaded = st.file_uploader("Input file (.csv or .xlsx)", type=["csv", "xlsx", "xls"], key="scrape_uploader")
    asof_default = datetime.now().isoformat(timespec="seconds")
    asof_str = st.text_input("As-of datetime (ISO8601)", value=asof_default, key="asof_input")

    run_btn = st.button("Run Scrape", type="primary", use_container_width=True)

    if run_btn:
        if not uploaded:
            st.error("Please upload an input file.")
            st.stop()

        # Read uploaded file into DataFrame
        try:
            df_in = load_uploaded_df(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        # Parse as-of
        try:
            asof_dt = asof_parse(asof_str)
        except Exception as e:
            st.error(f"Invalid 'as-of' datetime: {e}")
            st.stop()

        # Map columns (tolerant)
        colmap = {c.strip().lower(): c for c in df_in.columns}
        if not colmap:
            st.error("Uploaded file appears empty or has no columns.")
            st.stop()

        org_col = (
            colmap.get("org name")
            or colmap.get("org_name")
            or colmap.get("org")
            or list(df_in.columns)[0]
        )
        url_col = colmap.get("url") or (list(df_in.columns)[1] if len(df_in.columns) > 1 else org_col)

        rows_all: List[Dict[str, Any]] = []
        total = len(df_in) if len(df_in) > 0 else 1
        prog = st.progress(0.0, text="Scraping...")

        for i, (_, row) in enumerate(df_in.iterrows(), start=1):
            org = _clean_cell(row.get(org_col))
            src = _clean_cell(row.get(url_col))
            if not src:
                prog.progress(min(i / total, 1.0), text=f"Skipped blank URL ({i}/{total})")
                continue
            try:
                out_rows = ep.scrape_org(org, src, asof_dt)
                rows_all.extend(out_rows)
            except Exception as e:
                rows_all.append({
                    "Event Name": "",
                    "Event Organiser": org,
                    "Date": "",
                    "Day": "",
                    "Time": "",
                    "Location": "",
                    "Event URL": "",
                    "Source URL": src,
                    "Summary": f"ERROR: {e}",
                    "Scraped At": asof_dt.isoformat(),
                })
            prog.progress(min(i / total, 1.0), text=f"Processed {i}/{total}")

        # Build output DataFrame
        out_df = pd.DataFrame(rows_all, columns=ep.FINAL_COLS)
        st.session_state.results_df = out_df.copy()

        st.subheader("Results")
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"events_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        show_quick_metrics(out_df)


# =========================
# Tab: Chat with Results
# =========================
def _keyword_score(row_text: str, query: str) -> int:
    # Simple keyword count scoring (case-insensitive, whole-ish words)
    if not query.strip():
        return 0
    toks = [t for t in re.split(r"\W+", query.lower()) if t]
    if not toks:
        return 0
    text = row_text.lower()
    score = 0
    for t in toks:
        # weight exact token occurrences a bit more than substring
        score += text.count(f" {t} ")
        score += text.count(f"{t}") * 0.5
    return int(score)

def _row_to_compact_text(row: pd.Series) -> str:
    # Compact single-line representation used in prompts
    parts = []
    for k in ["Event Name","Event Organiser","Date","Day","Time","Location","Event URL","Source URL","Summary"]:
        v = str(row.get(k) or "").strip()
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)

def _top_rows_for_query(df: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    # Build a light searchable string per row
    texts = df.apply(_row_to_compact_text, axis=1)
    scores = texts.apply(lambda s: _keyword_score(s, query))
    top_idx = scores.sort_values(ascending=False).head(limit).index
    top_df = df.loc[top_idx].copy()
    # Keep only rows with positive score unless query blank
    if query.strip():
        top_df = top_df[scores.loc[top_idx] > 0]
    return top_df

def _llm_answer_from_rows(question: str, rows: pd.DataFrame) -> str:
    """Ask the LLM to answer strictly from the provided rows; no speculation."""
    from openai import OpenAI

    client = OpenAI()
    rows_json = rows.to_dict(orient="records")

    system = (
        "You are an assistant that answers strictly using the table rows provided as 'EVENT_ROWS'. "
        "Do not invent facts. If the answer is unclear from the rows, say what is missing. "
        "Be concise and helpful."
    )
    user = f"""
QUESTION:
{question}

EVENT_ROWS (JSON):
{json.dumps(rows_json, ensure_ascii=False)[:18000]}

INSTRUCTIONS:
- Cite event names and dates when making recommendations.
- If the user asks to filter (e.g., 'this Friday', 'GetWITit', 'after 5pm'), list matching rows succinctly.
- If nothing matches, say so and suggest the closest few rows from the provided data (do not use outside knowledge).
- Keep answers under ~150 words unless explicitly asked for more detail.
"""

    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_SUMMARY", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

with tab_chat:
    st.subheader("Chat with the current results")
    st.caption("Ask questions like: *“show networking events after 5pm this week”*, *“events by Ascender”*, or *“what’s happening on Sep 16?”*")

    # Optionally upload a CSV to chat with (instead of the latest scrape)
    with st.expander("Use a different CSV for chat (optional)"):
        chat_upload = st.file_uploader("Upload results CSV", type=["csv"], key="chat_csv")
        if chat_upload is not None:
            try:
                df_chat = pd.read_csv(chat_upload)
                # Validate columns
                missing = [c for c in ep.FINAL_COLS if c not in df_chat.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing columns: {missing}")
                else:
                    st.session_state.results_df = df_chat.copy()
                    st.success("Loaded CSV for chat.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    results_df = st.session_state.results_df.copy()

    # Optional quick filters to shrink the target table before chatting
    with st.expander("Quick filter (optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            org_filter = st.text_input("Filter by organiser contains", value="")
        with col2:
            date_from = st.text_input("Date from (YYYY-MM-DD)", value="")
        with col3:
            date_to = st.text_input("Date to (YYYY-MM-DD)", value="")

        df_view = results_df
        if org_filter.strip():
            df_view = df_view[df_view["Event Organiser"].astype(str).str.contains(org_filter, case=False, na=False)]
        if date_from.strip():
            try:
                df_view = df_view[df_view["Date"] >= date_from.strip()]
            except Exception:
                st.warning("Could not apply 'Date from' filter.")
        if date_to.strip():
            try:
                df_view = df_view[df_view["Date"] <= date_to.strip()]
            except Exception:
                st.warning("Could not apply 'Date to' filter.")

        st.dataframe(df_view, use_container_width=True, hide_index=True)
        show_quick_metrics(df_view)

    st.markdown("---")

    # Chat UI
    chat_container = st.container()
    with chat_container:
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        prompt = st.chat_input("Ask about the events…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Pick top rows by naive keyword scoring before LLM
            candidate_rows = _top_rows_for_query(results_df, prompt, limit=30)

            if api_key:
                try:
                    answer = _llm_answer_from_rows(prompt, candidate_rows)
                except Exception as e:
                    # Safe fallback if OpenAI call fails
                    answer = None
            else:
                answer = None

            if answer is None:
                # No key or LLM failed → rule-based fallback
                if candidate_rows.empty:
                    fallback = "I couldn’t find anything matching that in the current table."
                else:
                    # Build a concise, deterministic answer
                    head = candidate_rows.head(5)
                    lines = []
                    for _, r in head.iterrows():
                        line = f"- {r.get('Event Name','')} — {r.get('Date','')} {r.get('Time','')}".strip()
                        loc = str(r.get("Location","") or "").strip()
                        org = str(r.get("Event Organiser","") or "").strip()
                        if loc:
                            line += f" — {loc}"
                        if org:
                            line += f" (by {org})"
                        lines.append(line)
                    fallback = "Here are the top matches I found:\n" + "\n".join(lines)
                answer = fallback

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

            # Optional: show which rows were used (collapsed)
            with st.expander("Rows considered for this answer"):
                if candidate_rows.empty:
                    st.info("No matching rows selected.")
                else:
                    st.dataframe(candidate_rows, use_container_width=True, hide_index=True)
