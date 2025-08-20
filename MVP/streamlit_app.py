import os
import io
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil import parser as dateparser
from event_pipeline import scrape_org

st.set_page_config(page_title="InnovatePGH Event Scraper (Merged)", layout="wide")

st.title("InnovatePGH – Event Calendar Scraper")
st.caption("Upload a CSV/XLSX with columns: 'Org name', 'URL'. We'll scrape upcoming events (>= as-of) and write 150–200 word newsletter blurbs.")

asof_str = st.text_input("As-of datetime (ISO8601)", value=datetime.now().isoformat())
asof_dt = dateparser.parse(asof_str)

uploaded = st.file_uploader("Upload input file (.csv or .xlsx)", type=["csv","xlsx","xls"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    data = uploaded.read()
    buf = io.BytesIO(data)
    if suffix == ".csv":
        df = pd.read_csv(buf)
    else:
        df = pd.read_excel(buf)
    

    if st.button("Run Scrape"):
        results = []
        for _, row in df.iterrows():
            # tolerate case variants: 'Org name'/'org_name', 'URL'/'url'
            cols = {c.strip().lower(): c for c in df.columns}
            org_col = cols.get("org name") or cols.get("org_name") or cols.get("org") or list(df.columns)[0]
            url_col = cols.get("url") or list(df.columns)[1]
            org = str(row.get(org_col) or "").strip()
            url = str(row.get(url_col) or "").strip()
            if not url:
                continue
            with st.spinner(f"Scraping {org or url} …"):
                try:
                    rows = scrape_org(org, url, asof_dt)
                    results.extend(rows)
                except Exception as e:
                    results.append({
                        "Event Name": "",
                        "Event Organiser": org,
                        "Location": "",
                        "Time": "",
                        "Event URL": "",
                        "Source URL": url,
                        "Scraped At": asof_dt.isoformat(),
                        "Summary": f"ERROR: {e}"
                    })
        out_df = pd.DataFrame(results)
        st.success(f"Found {len(out_df)} event rows")
        st.dataframe(out_df, use_container_width=True)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="events_out.csv", mime="text/csv")
