import os
import io
from datetime import datetime

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser

# Local import: the pipeline script must be in the same folder
import event_pipeline as ep

st.set_page_config(page_title="InnovatePGH Event Scraper", layout="wide")

st.title("InnovatePGH – Event Scraper")

# ==== OpenAI API Key via Streamlit Secrets ====
# Priority: st.secrets -> environment -> (optional) text input
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

# Optional model overrides via secrets
if "OPENAI_MODEL_EXTRACT" in st.secrets:
    os.environ["OPENAI_MODEL_EXTRACT"] = st.secrets["OPENAI_MODEL_EXTRACT"]
if "OPENAI_MODEL_SUMMARY" in st.secrets:
    os.environ["OPENAI_MODEL_SUMMARY"] = st.secrets["OPENAI_MODEL_SUMMARY"]

# Allow user override (useful on local dev)
with st.expander("OpenAI Settings", expanded=False):
    manual_key = st.text_input("OpenAI API Key (overrides secrets for this session)", type="password")
    if manual_key:
        api_key = manual_key

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("OpenAI API key loaded.")
else:
    st.warning("OpenAI API key not found in Streamlit Secrets. Summaries/LLM extraction will use rule-based fallback.")

# ==== Input controls ====
st.markdown("Upload a CSV/XLSX with **Org name** and **URL** columns.")

uploaded = st.file_uploader("Input file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
asof_default = datetime.now().isoformat(timespec="seconds")
asof_str = st.text_input("As-of datetime (ISO8601)", value=asof_default)

run_btn = st.button("Run Scrape", type="primary", use_container_width=True)

# ==== Run pipeline ====
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
    url_col = colmap.get("url") or list(df_in.columns)[1]

    rows_all = []
    prog = st.progress(0.0, text="Scraping...")
    total = len(df_in)

    for i, (_, r) in enumerate(df_in.iterrows(), start=1):
        org = str(r.get(org_col) or "").strip()
        src = str(r.get(url_col) or "").strip()
        if not src:
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
        prog.progress(min(i/total, 1.0), text=f"Scraped {i}/{total}")

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
    st.caption(f"Rows: {len(out_df)} | With Date: {(out_df['Date'].astype(str).str.len()>0).sum()} | With URL: {(out_df['Event URL'].astype(str).str.len()>0).sum()}")
