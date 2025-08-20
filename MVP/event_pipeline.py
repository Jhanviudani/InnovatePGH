import os
import re
import json
import argparse
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# =========================
# Config
# =========================
USER_AGENT = "Mozilla/5.0 (compatible; InnovatePGH-EventBot/2.0)"
OPENAI_MODEL_EXTRACT = os.getenv("OPENAI_MODEL_EXTRACT", "gpt-4o-mini")   # small/fast for extraction
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")   # summary 150–200 words
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))

FINAL_COLS = [
    "Event Name", "Event Organiser", "Date", "Day", "Time",
    "Location", "Event URL", "Source URL", "Summary", "Scraped At",
]

# Minimal headings / boilerplate filters (keep small)
JUNK_TITLES = {
    "skip to main content", "upcoming events", "events", "month", "list",
    "view event", "view calendar", "previous events", "next events",
    "today", "search", "menu", "linkedin", "twitter", "instagram",
    "about", "sponsor", "loading view. events search and views navigation search enter keyword.",
}

# =========================
# Small helpers
# =========================
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_junk_title(title: str) -> bool:
    if not title:
        return True
    t = normalize_ws(title).lower()
    if len(t) < 4:
        return True
    if t in JUNK_TITLES:
        return True
    # Month heading like "September 2025"
    if re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", t):
        return True
    # calendar junk like "1 event, 9"
    if re.fullmatch(r"\d+\s+event[s]?,\s*\d+", t):
        return True
    if t.endswith("events") and ("upcoming" in t or t == "events"):
        return True
    return False

def guess_title_from_block(text: str) -> str:
    # crude but cheap: take the first sentence-ish fragment
    t = normalize_ws(text)
    m = re.split(r"[.\n]", t, maxsplit=1)
    return " ".join((m[0] if m else t).split()[:12])

# =========================
# Domain-aware link validation (keep thin but precise)
# =========================
def is_probable_event_link(abs_href: str, link_text: str) -> bool:
    if not abs_href:
        return False
    href = abs_href.strip()
    if href.startswith("#"):
        return False

    parsed = urlparse(href)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    text = normalize_ws(link_text).lower()

    # Global denies (listing/login/reset/navigation)
    deny_substrings = [
        "/events$", "/events/", "/events/list", "/events/month", "/calendar",
        "resetpassword", "login", "log-in", "logout", "/sys/", "viewmode",
        "search", "addtaganchorlink", "/sponsor", "#events", "/home",
    ]
    if any(ds in href.lower() for ds in deny_substrings):
        return False

    # Host‑specific rules for your current sources
    host_rules = {
        "community.pdma.org": {
            "allow": ["calendareventkey=", "/event-description"],
            "deny": ["addtaganchorlink", "/home", "/events/"],
        },
        "ascenderpgh.com": {
            "allow": ["/event/"],
            "deny": ["/events/list/"],
        },
        "www.robopgh.org": {
            "allow": ["/robotics-discovery-day", "eventbrite.com/e"],
            "deny": ["/events", "/sponsor"],
        },
        "robopgh.org": {
            "allow": ["/robotics-discovery-day", "eventbrite.com/e"],
            "deny": ["/events", "/sponsor"],
        },
        "www.bigidea.pitt.edu": {
            "allow": ["/event/"],
            "deny": ["/events/month", "/events/20", "/events/"],  # calendar/list pages
        },
        "www.pghtech.org": {
            "allow": ["/events/"],  # detail pages under /events/<slug>
            "deny": ["/events$"],
        },
        "amapittsburgh.org": {
            "allow": ["/event/"],
            "deny": ["/events/"],
        },
        "pyp23.wildapricot.org": {
            "allow": ["/event-"],   # detail is /event-<id>
            "deny": ["resetpassword", "/sys/", "/events", "viewmode", "registration"],
        },
        "getwitit.org": {
            "allow": ["eventbrite.com/e"],
            "deny": ["/chapter/", "/chapter/pittsburgh-chapter/"],
        },
        "eventbrite.com": {
            "allow": ["/e/"],
            "deny": [],
        }
    }

    if host in host_rules:
        allow, deny = host_rules[host]["allow"], host_rules[host]["deny"]
        if any(d in href.lower() for d in deny):
            return False
        if any(a in href.lower() for a in allow):
            return True

    # Generic allow: detail-y paths
    if "/event/" in path or "calendareventkey=" in href.lower() or "eventbrite.com/e" in href.lower():
        return True

    # Last resort: path contains "event" and non-junk link text
    if "event" in path and len(text) >= 6 and not is_junk_title(text):
        return True

    return False

# =========================
# HTTP + soup
# =========================
def fetch(url: str, timeout: int = 30):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text, resp.url

# =========================
# LLM extraction (JSON mode)
# =========================
def llm_extract_event_fields(context_text: str, page_url: str, org_name: Optional[str]) -> Dict[str, Any]:
    """
    Use LLM to extract a single event from noisy context.
    Returns:
      event_name, date_iso, day_name, time_text, location, event_url, confidence
    Any field may be None if unclear. No guessing.
    """
    if not USE_LLM:
        return {}

    from openai import OpenAI
    client = OpenAI()

    system = (
        "You extract event metadata from noisy web text. "
        "Return STRICT JSON. If a field is unclear, set it to null. Do not invent."
    )

    schema_hint = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date_iso":   {"type": "string", "description": "YYYY-MM-DD if clear; else null"},
            "day_name":   {"type": "string", "description": "Monday..Sunday if date exists; else null"},
            "time_text":  {"type": "string"},
            "location":   {"type": "string"},
            "event_url":  {"type": "string", "description": "Detail page URL if present; else null"},
            "confidence": {"type": "number"}
        },
        "required": ["event_name","date_iso","day_name","time_text","location","event_url","confidence"],
        "additionalProperties": False
    }

    user = f"""
Extract a single future (or same-day) event from the text below.

HARD RULES:
- Do NOT fabricate. If unclear, use null.
- Prefer event detail URLs over listing/login/reset pages. If only a listing is present, set event_url=null.
- Title must be clean (no 'Starts:', 'Select date', 'View Event', headings, or month-only strings).
- If multiple dates appear (range), use the first start date.
- Output JSON ONLY and validate the schema: {json.dumps(schema_hint)}

PAGE_URL: {page_url}
ORGANIZER (hint): {org_name or ""}

TEXT:
---
{context_text[:3500]}
---
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_EXTRACT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return {}

    # Light normalization
    for k in ["event_name","date_iso","day_name","time_text","location","event_url"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip() or None
    try:
        data["confidence"] = float(data.get("confidence", 0.0))
    except Exception:
        data["confidence"] = 0.0
    if data["confidence"] < 0: data["confidence"] = 0.0
    if data["confidence"] > 1: data["confidence"] = 1.0
    return data

# =========================
# Summary (150–200 words)
# =========================
def summarize(name: str, org: Optional[str], raw_text: str) -> str:
    if not USE_LLM:
        # rule-based fallback
        base = normalize_ws(raw_text)
        snippet = " ".join(base.split()[:120])
        parts = []
        if name: parts.append(f"{name} is an upcoming event")
        if org: parts.append(f"hosted by {org}")
        parts.append("for Pittsburgh’s startup and tech community.")
        if snippet: parts.append(f"Highlights: {snippet}.")
        return normalize_ws(" ".join(parts))

    from openai import OpenAI
    client = OpenAI()
    prompt = f"""
You are an expert newsletter editor. Craft ONE polished paragraph of ~150–200 words (no bullets) for an event.

Requirements:
- Open with what the event is and who it is for.
- Mention organizer "{org or 'Unknown'}" succinctly.
- Describe core topic, expected format (panel/workshop/networking), and practical takeaways.
- If time/place are clearly present, include them briefly; if unclear, omit rather than guess.
- Avoid hype, clichés, and exclamation marks.
- Paraphrase—do NOT copy phrasing from the source.

Event name: "{name or 'Unknown'}"
Raw context (may be noisy; deduplicate ideas): {raw_text[:2000]}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_SUMMARY,
        messages=[
            {"role":"system","content":"You write precise, concise, plagiarism-free newsletter blurbs."},
            {"role":"user","content":prompt}
        ],
        temperature=0.3,
        max_tokens=480
    )
    return normalize_ws(resp.choices[0].message.content.strip())

# =========================
# Extraction flow
# =========================
def extract_candidates(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Find likely event-detail anchors; return (title, url, context_text)."""
    candidates = []
    for a in soup.find_all("a"):
        link_text = a.get_text(" ", strip=True) or ""
        href = a.get("href")
        if not href:
            continue
        abs_href = urljoin(base_url, href)
        if not is_probable_event_link(abs_href, link_text):
            continue

        # try a small container to gather context
        node = a
        for _ in range(3):
            if node and node.parent:
                node = node.parent
        context_text = node.get_text(" ", strip=True) if node else link_text

        candidates.append({
            "title": link_text,
            "url": abs_href,
            "context": context_text[:4000],
        })
    return candidates

def build_row_from_llm(block: Dict[str, str], org_name: str, asof_dt: datetime, source_url: str) -> Optional[Dict[str, str]]:
    data = llm_extract_event_fields(block["context"], block["url"], org_name)
    # If LLM unavailable or empty → fallback
    if not data or not data.get("event_name"):
        # Tiny heuristic fallback
        name = guess_title_from_block(block["context"])
        if is_junk_title(name):
            return None
        dt = None
        try:
            # try to pull a date string from context (very light)
            m = re.search(r"(?i)(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})", block["context"])
            if m:
                dt = dateparser.parse(m.group(0))
        except Exception:
            dt = None
        if dt and dt.date() < asof_dt.date():
            return None
        return {
            "Event Name": name,
            "Event Organiser": org_name,
            "Date": dt.strftime("%Y-%m-%d") if dt else "",
            "Day": dt.strftime("%A") if dt else "",
            "Time": (re.search(r"(?i)\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", block["context"]) or re.search(r"\b\d{2}:\d{2}\b", block["context"]) or [None]).group(0) if re.search(r"(?i)\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", block["context"]) or re.search(r"\b\d{2}:\d{2}\b", block["context"]) else "",
            "Location": "",  # keep light in fallback
            "Event URL": block["url"],
            "Source URL": source_url,
            "Summary": summarize(name, org_name, block["context"]),
            "Scraped At": asof_dt.isoformat(),
        }

    # Validate LLM output
    name = data.get("event_name")
    if is_junk_title(name):
        return None

    # Date
    dt = dateparser.parse(data["date_iso"]) if data.get("date_iso") else None
    if dt and dt.date() < asof_dt.date():
        return None

    # require detail URL or date
    ev_url = data.get("event_url")
    if not ev_url and not dt:
        return None

    # day normalization
    day_name = data.get("day_name") or (dt.strftime("%A") if dt else "")
    if dt and day_name and day_name != dt.strftime("%A"):
        day_name = dt.strftime("%A")

    return {
        "Event Name": name,
        "Event Organiser": org_name,
        "Date": dt.strftime("%Y-%m-%d") if dt else "",
        "Day": day_name or "",
        "Time": data.get("time_text") or "",
        "Location": data.get("location") or "",
        "Event URL": ev_url or "",
        "Source URL": source_url,
        "Summary": summarize(name, org_name, block["context"]),
        "Scraped At": asof_dt.isoformat(),
    }

# =========================
# Main scrape for an org
# =========================
def scrape_org(org_name: str, url: str, asof_dt: datetime) -> List[Dict[str, str]]:
    html, final_url = fetch(url)
    soup = BeautifulSoup(html, "lxml")

    # 1) find likely event detail anchors
    blocks = extract_candidates(soup, final_url)

    # 2) LLM extract (or fallback) per block
    rows: List[Dict[str, str]] = []
    seen = set()
    for b in blocks:
        row = build_row_from_llm(b, org_name, asof_dt, final_url)
        if not row:
            continue
        key = (row["Event Name"].lower(), row["Date"], row["Event URL"].lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    return rows

# =========================
# IO
# =========================
def read_input(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(description="InnovatePGH Event Scraper – LLM Hybrid")
    parser.add_argument("--input", required=True, help="CSV/XLSX with columns: 'Org name', 'URL'")
    parser.add_argument("--output", default="events_out.csv", help="Output CSV path")
    parser.add_argument("--asof", default=datetime.now().isoformat(), help="ISO datetime (filters to future/same-day)")
    args = parser.parse_args()

    asof_dt = dateparser.parse(args.asof)
    df = read_input(args.input)

    # Accept 'Org name','URL' (case tolerant)
    colmap = {c.strip().lower(): c for c in df.columns}
    org_col = colmap.get("org name") or colmap.get("org_name") or colmap.get("org") or list(df.columns)[0]
    url_col = colmap.get("url") or list(df.columns)[1]

    all_rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        org = str(row.get(org_col) or "").strip()
        src = str(row.get(url_col) or "").strip()
        if not src:
            continue
        try:
            out = scrape_org(org, src, asof_dt)
            all_rows.extend(out)
        except Exception as e:
            all_rows.append({
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

    out_df = pd.DataFrame(all_rows, columns=FINAL_COLS)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} rows to {args.output}")

if __name__ == "__main__":
    main()
