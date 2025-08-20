# InnovatePGH Event Scraper (Merged)

**Input**: CSV/XLSX with columns **'Org name', 'URL'**  
**Output** (CSV): 
- Event Name
- Event Organiser
- Date (YYYY-MM-DD)
- Day (Monday–Sunday)
- Time (e.g., 5:30 PM)
- Location
- Event URL
- Source URL
- Summary (150–200 words, newsletter-ready)
- Scraped At (ISO8601) — last column

## Run (CLI)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...    # optional; enables high-quality 150–200 word summaries
export OPENAI_MODEL=gpt-4o-mini # optional

python event_pipeline.py --input sample_input.csv --output events_out.csv
# or set the filter date explicitly
python event_pipeline.py --input sample_input.csv --asof "2025-08-20T09:00:00" --output events_out.csv
```

## Run (Streamlit)

```bash
streamlit run streamlit_app.py
```

Upload your sheet and click **Run Scrape**.

## Notes
- Filters out nav/heading junk (e.g., “Skip to main content”, month headings).
- Prefers **schema.org/Event** JSON‑LD; falls back to **heuristics**.
- Keeps only events **on/after** the **as-of** date (defaults to now).
- If no `OPENAI_API_KEY`, summaries fall back to a rule-based version.
