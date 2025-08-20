import os
import re
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# =========================
# Config
# =========================
USER_AGENT = "Mozilla/5.0 (compatible; InnovatePGH-EventBot/2.2)"
OPENAI_MODEL_EXTRACT = os.getenv("OPENAI_MODEL_EXTRACT", "gpt-4o-mini")   # small/fast for extraction
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")   # 150–200 words
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
    "complete this sponsorship form.",
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

def http_get(url: str, timeout: int = 30):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text, r.url

# =========================
# Domain-aware link validation (thin but precise)
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
    q = (parsed.query or "").lower()
    text = normalize_ws(link_text).lower()

    # Global denies (listing/login/reset/navigation/resources/waivers/guidelines/surveys)
    deny_substrings = [
        "/events$", "/events/", "/calendar", "/events/list", "/events/month",
        "resetpassword", "login", "log-in", "logout", "/sys/", "viewmode",
        "search", "addtaganchorlink", "/sponsor", "#events", "/home",
        "/resources", "/resource", "/event-space", "/event-waiver",
        "surveymonkey.com", "feedback", "guidelines", "pyp-event-guidelines",
        "/chapters/", "/chapter/", "/pittsburgh-business-events"
    ]
    if any(ds in href.lower() for ds in deny_substrings):
        return False

    # Host‑specific rules
    host_rules = {
        "community.pdma.org": {
            "allow": ["calendareventkey=", "/event-description"],
            "deny": ["addtaganchorlink", "/home", "/events/"],
        },
        "ascenderpgh.com":       {"allow": ["/event/"], "deny": ["/events/list/","/event-space/"]},
        "www.robopgh.org":       {"allow": ["/robotics-discovery-day", "eventbrite.com/e"], "deny": ["/events", "/sponsor"]},
        "robopgh.org":           {"allow": ["/robotics-discovery-day", "eventbrite.com/e"], "deny": ["/events", "/sponsor"]},
        "www.bigidea.pitt.edu":  {"allow": ["/event/"], "deny": ["/events/month", "/events/20", "/events/"]},
        "www.pghtech.org":       {"allow": ["/events/"], "deny": ["/events$"]},
        "amapittsburgh.org":     {"allow": ["/event/"], "deny": ["/events/"]},
        "pyp23.wildapricot.org": {"allow": ["/event-"], "deny": ["resetpassword", "/sys/", "/events", "viewmode", "registration", "guidelines"]},
        "getwitit.org":          {"allow": ["eventbrite.com/e"], "deny": ["/chapter/", "/chapters/"]},
        "eventbrite.com":        {"allow": ["/e/"], "deny": []},
        "bridgecityconnections.com": {"allow": [], "deny": ["/pittsburgh-business-events"]},  # aggregator page only
        "ellevatenetwork.com":   {"allow": ["/events/"], "deny": ["/events?$","/chapters/","/event-waiver"]},
    }

    if host in host_rules:
        allow, deny = host_rules[host]["allow"], host_rules[host]["deny"]
        if any(d in href.lower() for d in deny):
            return False
        if allow and any(a in href.lower() for a in allow):
            return True

    # Meetup guard: only accept Pittsburgh groups (e.g., /producttank-pittsburgh/, /code-and-coffee-pgh/)
    if host.endswith("meetup.com"):
        # Require a group path containing "pittsburgh" or "pgh"
        if not re.search(r"(pittsburgh|pgh)", path):
            return False
        # detail links include /events/<id>
        if "/events/" not in path:
            return False
        return True

    # Generic allow: detail-y paths
    if "/event/" in path or "calendareventkey=" in href.lower() or "eventbrite.com/e" in href.lower():
        return True

    # Last resort: path contains "event" and non-junk link text
    if "event" in path and len(text) >= 6 and not is_junk_title(text):
        return True

    return False

# =========================
# Listing → candidates
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

# =========================
# Detail page extraction
# =========================
def extract_json_ld_events(soup: BeautifulSoup, base_url: str):
    out = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string) if script.string else None
        except Exception:
            data = None
        if not data:
            continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            if not isinstance(b, dict):
                continue
            types = b.get("@type")
            types = types if isinstance(types, list) else [types]
            if "Event" not in [t for t in types if t]:
                continue
            url = b.get("url")
            url = urljoin(base_url, url) if url else None
            loc_name = None
            loc_obj = b.get("location") or {}
            if isinstance(loc_obj, dict):
                loc_name = loc_obj.get("name")
                addr = loc_obj.get("address")
                if isinstance(addr, dict):
                    parts = [addr.get("streetAddress"), addr.get("addressLocality"),
                             addr.get("addressRegion"), addr.get("postalCode")]
                    loc_name = loc_name or ", ".join([p for p in parts if p])
                elif isinstance(addr, str):
                    loc_name = loc_name or addr
            out.append({
                "name": (b.get("name") or "").strip() or None,
                "start": b.get("startDate") or b.get("startTime"),
                "end": b.get("endDate") or b.get("endTime"),
                "location": loc_name or None,
                "url": url,
                "desc": b.get("description") or "",
            })
    return out

def extract_title_h1(soup: BeautifulSoup):
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return None

def page_plain_text(soup: BeautifulSoup, limit: int = 12000):
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)[:limit]

def build_event_from_detail(detail_url: str, org_name: str, asof_dt: datetime, source_url_for_row: str):
    try:
        html, final_url = http_get(detail_url)
    except Exception:
        return None

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "")[:140].lower()
    if any(x in title for x in ["waiver","guidelines","feedback","survey","resources","resource","event space","our spaces"]):
        return None

    # 1) schema.org first
    ld_events = extract_json_ld_events(soup, final_url)
    if ld_events:
        b = ld_events[0]  # usually one event per detail page
        name = b["name"] or extract_title_h1(soup)
        dt = dateparser.parse(b["start"]) if b.get("start") else None
        if dt and dt.date() < asof_dt.date():
            return None
        return {
            "Event Name": name or "",
            "Event Organiser": org_name,
            "Date": dt.strftime("%Y-%m-%d") if dt else "",
            "Day": dt.strftime("%A") if dt else "",
            "Time": "",  # can be derived from dt if present; leave blank if not clear
            "Location": b.get("location") or "",
            "Event URL": b.get("url") or final_url,
            "Source URL": source_url_for_row,
            "Summary": "",  # fill later (LLM)
            "Scraped At": asof_dt.isoformat(),
            "_context": (b.get("desc","") or "") + " " + page_plain_text(soup),
        }

    # 2) fallback: h1/title + page text + light date parse
    name = extract_title_h1(soup)
    if not name or is_junk_title(name):
        return None

    whole = page_plain_text(soup)
    # very light date parse from full page
    m = re.search(r"(?i)(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})", whole)
    dt = dateparser.parse(m.group(0)) if m else None
    if dt and dt.date() < asof_dt.date():
        return None

    tm = (re.search(r"(?i)\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", whole) or re.search(r"\b\d{2}:\d{2}\b", whole))
    loc = None
    loc_hint = re.search(r"(?i)(location\s*[:\-]\s*)([^|\n]+)", whole)
    if loc_hint:
        loc = loc_hint.group(2).strip()

    return {
        "Event Name": name,
        "Event Organiser": org_name,
        "Date": dt.strftime("%Y-%m-%d") if dt else "",
        "Day": dt.strftime("%A") if dt else "",
        "Time": tm.group(0) if tm else "",
        "Location": loc or "",
        "Event URL": final_url,
        "Source URL": source_url_for_row,
        "Summary": "",
        "Scraped At": asof_dt.isoformat(),
        "_context": whole,
    }

# =========================
# LLM extraction (JSON mode) + Summary
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

def summarize(name: str, org: Optional[str], raw_text: str) -> str:
    if not USE_LLM:
        base = normalize_ws(raw_text)
        # take first meaningful ~60–80 words from body (skip nav words)
        sentences = re.split(r"(?<=[.!?])\s+", base)
        body = " ".join([s for s in sentences if len(s.split()) > 6][:5])  # ~5 decent sentences
        opener = f"{name} is a community event" if not org else f"{name} is a community event hosted by {org}"
        return normalize_ws(f"{opener}. {body}")[:1100]

    from openai import OpenAI
    client = OpenAI()
    prompt = f"""
You are an expert newsletter editor. Write ONE paragraph of ~150–200 words (no bullets, no headings, no 'Highlights:').

Cover:
- What the event is and who it’s for.
- Organizer "{org or 'Unknown'}".
- Format (panel/workshop/networking) and practical takeaways.
- If time/location are clearly present, include briefly; if unclear, omit (do not invent).
- Paraphrase and keep it concise.

Event name: "{name or 'Unknown'}"
Source text (deduped body): {raw_text[:1800]}
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
# Scrape an org: listing → candidates → detail pages → LLM finalize
# =========================
def scrape_org(org_name: str, url: str, asof_dt: datetime) -> List[Dict[str, str]]:
    html, final_url = http_get(url)
    soup = BeautifulSoup(html, "lxml")

    # 1) find likely event-detail anchors
    blocks = extract_candidates(soup, final_url)

    # 2) visit each detail page and build a record
    rows: List[Dict[str, str]] = []
    seen = set()
    for b in blocks:
        detail_row = build_event_from_detail(b["url"], org_name, asof_dt, source_url_for_row=final_url)
        if not detail_row:
            continue
        key = (detail_row["Event Name"].lower(), detail_row["Date"], detail_row["Event URL"].lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(detail_row)

    # 3) finalize fields via LLM (if available) and add 150–200 word summary
    finalized: List[Dict[str, str]] = []
    seen2 = set()
    for r in rows:
        if USE_LLM:
            data = llm_extract_event_fields(r.get("_context", ""), r["Event URL"], r["Event Organiser"])
            # merge only when present
            if data.get("event_name") and not is_junk_title(data["event_name"]):
                r["Event Name"] = data["event_name"]
            if data.get("date_iso"):
                try:
                    dt = dateparser.parse(data["date_iso"])
                    if dt and dt.date() >= asof_dt.date():
                        r["Date"] = dt.strftime("%Y-%m-%d")
                        r["Day"]  = dt.strftime("%A")
                except Exception:
                    pass
            if data.get("time_text"): r["Time"] = data["time_text"]
            if data.get("location"):  r["Location"] = data["location"]
            if data.get("event_url"): r["Event URL"] = data["event_url"]

        # Summary based on detail page context
        ctx = r.pop("_context", "")
        r["Summary"] = summarize(r["Event Name"], r["Event Organiser"], ctx)

        # Drop past events (last guard) and enforce minimal validity
        if r["Date"]:
            try:
                dtt = dateparser.parse(r["Date"])
                if dtt and dtt.date() < asof_dt.date():
                    continue
            except Exception:
                pass

        key2 = (r["Event Name"].lower(), r["Date"], r["Event URL"].lower())
        if key2 in seen2:
            continue
        seen2.add(key2)
        finalized.append(r)

    return finalized

def _strip_boilerplate(text: str) -> str:
    # Remove common chrome fragments that poison summaries
    junk_patterns = [
        r"Skip to main content.*?",
        r"Toggle navigation",
        r"\b(About|Events|Programs|Membership|Donate|Contact)\b(?![\w-])",
        r"Login|Log In|Sign In|Sign Up",
        r"Add to calendar.*?(Google Calendar|iCalendar|Outlook)",
        r"Cookie(s)? Policy|Privacy|Terms|Accessibility",
        r"Find my tickets|Eventbrite.*?(Find Events|Create Events|Help Center)",
        r"Menu Close",
    ]
    t = text
    for pat in junk_patterns:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    # de-duplicate long repeated sequences
    t = re.sub(r"\s+", " ", t).strip()
    return t

def page_plain_text(soup: BeautifulSoup, limit: int = 10000):
    # remove obvious non-content nodes
    for sel in ["script","style","noscript","header","footer","nav","form","aside"]:
        for tag in soup.find_all(sel):
            tag.decompose()
    # favor a main content container if present
    main = soup.find("main") or soup.find(attrs={"role":"main"}) or soup.body
    text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
    text = _strip_boilerplate(text)
    return text[:limit]


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
    parser = argparse.ArgumentParser(description="InnovatePGH Event Scraper – Detail + LLM Hybrid")
    parser.add_argument("--input", required=True, help="CSV/XLSX with columns: 'Org name', 'URL'")
    parser.add_argument("--output", default="events_out.csv", help="Output CSV path")
    parser.add_argument("--asof", default=datetime.now().isoformat(), help="ISO datetime (filters to future/same-day)")
    args = parser.parse_args()

    asof_dt = dateparser.parse(args.asof)
    df = read_input(args.input)

    # Accept 'Org name','URL' (case tolerant)
    colmap = {c.strip().lower(): c for c in df.columns}
    org_col = colmap.get("org name") or colmap.get("org_name") or colmap.get("org")


