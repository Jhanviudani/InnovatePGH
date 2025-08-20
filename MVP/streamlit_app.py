import os
from datetime import datetime

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser

# Local import: must be in the same folder with this exact name
import event_pipeline as ep

st.set_page_config(page_title="InnovatePGH Event Scraper", layout="wide")
st.title("InnovatePGH – Event Scraper")

# ==== OpenAI API Key via Streamlit Secrets ====
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("OpenAI API key loaded from secrets. LLM extraction + summaries ENABLED.")
    st.caption(f"Models: extract={os.environ.get('OPENAI_MODEL_EXTRACT','gpt-4o-mini')}, summary={os.environ.get('OPENAI_MODEL_SUMMARY','gpt-4o-mini')}")
else:
    st.warning("No API key found in Streamlit secrets. Fallback heuristics will be used (summaries generic).")

# Optional model overrides via secrets
if "OPENAI_MODEL_EXTRACT" in st.secrets:
    os.environ["OPENAI_MODEL_EXTRACT"] = st.secrets["OPENAI_MODEL_EXTRACT"]
if "OPENAI_MODEL_SUMMARY" in st.secrets:
    os.environ["OPENAI_MODEL_SUMMARY"] = st.secrets["OPENAI_MODEL_SUMMARY"]

# Allow manual override (useful for local dev)
with st.expander("OpenAI Settings", expanded=False):
    manual_key = st.text_input("OpenAI API Key (overrides secrets for this session)", type="password")
    if manual_key:
        api_key = manual_key
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("OpenAI API key loaded for this session.")

# ==== Input controls ====
st.markdown("Upload a CSV/XLSX with **Org name** and **URL** columns.")

uploaded = st.file_uploader("Input file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
asof_default = datetime.now().isoformat(timespec="seconds")
asof_str = st.text_input("As-of datetime (ISO8601)", value=asof_default)

run_btn = st.button("Run Scrape", type="primary", use_container_width=True)

def _clean_cell(val) -> str:
    # Convert NaN/None to "", and "nan" (string) to ""
    if isinstance(val, str):
        s = val.strip()
        return "" if s.lower() == "nan" else s
    if pd.isna(val):
        return ""
    return str(val).strip()

if run_btn:
    if not uploaded:
        st.error("Please upload an input file.")
        st.stop()

    # Read uploaded file into DataFrame
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    try:
        asof_dt = dateparser.parse(asof_str)
    except Exception as e:
        st.error(f"Invalid 'as-of' datetime: {e}")
        st.stop()

    # Map columns (tolerant)
    colmap = {c.strip().lower(): c for c in df_in.columns}
    org_col = colmap.get("org name") or colmap.get("org_name") or colmap.get("org") or list(df_in.columns)[0]
    url_col = colmap.get("url") or (list(df_in.columns)[1] if len(df_in.columns) > 1 else org_col)

    rows_all = []
    prog = st.progress(0.0, text="Scraping...")
    total = len(df_in) if len(df_in) > 0 else 1

    for i, (_, r) in enumerate(df_in.iterrows(), start=1):
        org = _clean_cell(r.get(org_col))
        src = _clean_cell(r.get(url_col))
        if not src:
            # skip blanks and 'nan'
            prog.progress(min(i/total, 1.0), text=f"Skipped blank URL ({i}/{total})")
            continue
        try:
            out = ep.scrape_org(org, src, asof_dt)
            rows_all.extend(out)
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
        prog.progress(min(i/total, 1.0), text=f"Processed {i}/{total}")

    # Build output DataFrame
    out_df = pd.DataFrame(rows_all, columns=ep.FINAL_COLS)

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

    # Quick metrics
    with_date = (out_df["Date"].astype(str).str.len() > 0).sum() if not out_df.empty else 0
    with_url = (out_df["Event URL"].astype(str).str.len() > 0).sum() if not out_df.empty else 0
    st.caption(f"Rows: {len(out_df)} | With Date: {with_date} | With URL: {with_url}")
