"""
MTMS — Micro-Trade Micro-Shock Detection Pipeline
Y2AI Research | Version 4.0 | March 2026

Folder layout:
    ontology/
        mtms.py
        ontology.json
        requirements.txt
        .env

Setup:
    pip install -r requirements.txt

Usage:
    python mtms.py --schema        print Postgres DDL (paste into Supabase SQL editor)
    python mtms.py --run           single pipeline pass
    python mtms.py --review        interactive Claude review
    python mtms.py --alerts        print pending alerts
    python mtms.py --log           print event log
    python mtms.py --report        feed quality report (items/day, relevance)
    python mtms.py --loop          run continuously

Cron (market hours every 5 minutes):
    */5 9-16 * * 1-5 cd /path/to/ontology && python mtms.py --run
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import feedparser
import httpx
import trafilatura
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from supabase import create_client

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))
load_dotenv()  # also check cwd

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "").strip()
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
ONTOLOGY_PATH     = os.getenv("ONTOLOGY_PATH", os.path.join(_SCRIPT_DIR, "ontology.json"))
SUPABASE_URL      = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY      = os.getenv("SUPABASE_KEY", "").strip()

if not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing ANTHROPIC_API_KEY in .env")
if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────
# FEEDS — Signal vs Discovery tiers
# ─────────────────────────────────────────────────────────────
# Tier A (signal=True)  → full pipeline + alert if score >= threshold
# Tier B (signal=False) → ingest + LLM only, no Polygon unless tradeable=True + confidence>0.7

FEEDS = [
    # ── Tier A: Vendor official blogs (direct, highest signal) ────
    {"url": "https://www.anthropic.com/news/rss.xml",                                                          "name": "Anthropic News",          "tier": "A"},
    {"url": "https://openai.com/blog/rss.xml",                                                                 "name": "OpenAI Blog",             "tier": "A"},
    {"url": "https://blog.google/technology/ai/rss/",                                                          "name": "Google AI Blog",          "tier": "A"},
    {"url": "https://blogs.microsoft.com/ai/feed/",                                                            "name": "Microsoft AI Blog",       "tier": "A"},
    {"url": "https://aws.amazon.com/blogs/aws/feed/",                                                          "name": "AWS Blog",                "tier": "A"},
    {"url": "https://ai.meta.com/blog/rss.xml",                                                                "name": "Meta AI Blog",            "tier": "A"},
    {"url": "https://feeds.feedburner.com/TheHackersNews",                                                     "name": "Hacker News Security",    "tier": "A"},

    # ── Tier A: Google Alerts — vendor capability releases ────────
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/10729782878002373362", "name": "GA: Anthropic releases",      "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/15834280160766176159", "name": "GA: OpenAI releases",         "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/1413566557852832786",  "name": "GA: Google/Gemini releases",  "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/4592139893980702595",  "name": "GA: Microsoft/Copilot",       "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/4592139893980700877",  "name": "GA: AWS Bedrock releases",    "tier": "A"},

    # ── Tier A: Google Alerts — capability classes (cross-vendor) ─
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/4592139893980701163",  "name": "GA: AI agents",               "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/7397045934501155708",  "name": "GA: Code/AppSec AI",          "tier": "A"},
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/4190648198015428629",  "name": "GA: Reasoning/multimodal",    "tier": "A"},

    # ── Tier B: Discovery — log only, no alert ────────────────────
    {"url": "https://www.google.com/alerts/feeds/11684174711489635674/12075800150936424031", "name": "GA: Discovery",               "tier": "B"},
]

# ─────────────────────────────────────────────────────────────
# SCORING THRESHOLDS  (calibrate after 20 events)
# ─────────────────────────────────────────────────────────────

THRESHOLDS = {
    "late_vol_mult":       2.5,
    "vwap_delta_pct":     -0.40,
    "trade_count_mult":    2.0,
    "close_position":      0.25,
    "min_score_to_alert":  2,
    "min_confidence":      0.60,
}

FETCH_INTERVAL = 300  # seconds between loop passes

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("mtms.log"), logging.StreamHandler()]
)
log = logging.getLogger("mtms")

# ─────────────────────────────────────────────────────────────
# POSTGRES DDL + VIEWS  (paste into Supabase SQL editor)
# ─────────────────────────────────────────────────────────────

MTMS_SCHEMA_SQL = """
-- ============================================================
-- MTMS tables
-- ============================================================

CREATE TABLE IF NOT EXISTS mtms_events (
    id              BIGSERIAL PRIMARY KEY,
    url             TEXT NOT NULL,
    url_hash        VARCHAR(64) UNIQUE NOT NULL,
    title           TEXT,
    title_hash      VARCHAR(64),
    source          VARCHAR(255),
    feed_tier       VARCHAR(4) DEFAULT 'A',
    published_at    VARCHAR(64),
    fetched_at      VARCHAR(64) NOT NULL,
    full_text       TEXT,
    status          VARCHAR(32) DEFAULT 'new',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mtms_events_title_hash ON mtms_events(title_hash);
CREATE INDEX IF NOT EXISTS idx_mtms_events_status ON mtms_events(status);

CREATE TABLE IF NOT EXISTS mtms_extractions (
    id                   BIGSERIAL PRIMARY KEY,
    event_id             BIGINT REFERENCES mtms_events(id),
    tradeable            SMALLINT,
    shock_type           VARCHAR(64),
    budget_line          TEXT,
    capability_ids       TEXT,
    bucket_a             TEXT,
    bucket_b             TEXT,
    sympathy_tickers     TEXT,
    reverse_tickers      TEXT,
    function_scores      TEXT,
    signal_quality       VARCHAR(16),
    monday_plan          TEXT,
    stand_down           TEXT,
    confidence           REAL,
    raw_json             TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mtms_scores (
    id               BIGSERIAL PRIMARY KEY,
    event_id         BIGINT REFERENCES mtms_events(id),
    ticker           VARCHAR(16) NOT NULL,
    bucket           VARCHAR(4),
    event_date       VARCHAR(16),
    price_move_pct   REAL,
    late_vol_mult    REAL,
    vwap_delta_pct   REAL,
    trade_count_mult REAL,
    close_position   REAL,
    score            INT,
    label            VARCHAR(64),
    monday_outcome   VARCHAR(64),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mtms_alerts (
    id         BIGSERIAL PRIMARY KEY,
    event_id   BIGINT REFERENCES mtms_events(id),
    score_id   BIGINT REFERENCES mtms_scores(id),
    ticker     VARCHAR(16),
    label      VARCHAR(64),
    sent_at    VARCHAR(64),
    channel    VARCHAR(32) DEFAULT 'console'
);

CREATE TABLE IF NOT EXISTS mtms_feed_health (
    id              BIGSERIAL PRIMARY KEY,
    feed_name       VARCHAR(255),
    feed_url        TEXT,
    feed_tier       VARCHAR(4),
    checked_at      TIMESTAMPTZ DEFAULT NOW(),
    entry_count     INT,
    status          VARCHAR(16),
    last_entry_title TEXT
);

-- ============================================================
-- Views for JOIN queries (used by --review, --alerts, --log, etc.)
-- ============================================================

CREATE OR REPLACE VIEW v_mtms_pending_review AS
SELECT e.id, e.title, e.url, e.full_text, e.source, e.feed_tier
FROM mtms_events e
LEFT JOIN mtms_extractions ex ON ex.event_id = e.id
WHERE e.full_text IS NOT NULL
  AND e.status = 'fetched'
  AND ex.id IS NULL
ORDER BY e.fetched_at DESC;

CREATE OR REPLACE VIEW v_mtms_pending_classification AS
SELECT e.id, e.title, e.full_text, e.feed_tier
FROM mtms_events e
LEFT JOIN mtms_extractions ex ON ex.event_id = e.id
WHERE e.full_text IS NOT NULL
  AND e.status = 'fetched'
  AND ex.id IS NULL;

CREATE OR REPLACE VIEW v_mtms_scoreable_events AS
SELECT e.id, e.published_at, e.fetched_at, e.feed_tier,
       ex.bucket_a, ex.bucket_b, ex.confidence
FROM mtms_events e
JOIN mtms_extractions ex ON ex.event_id = e.id
LEFT JOIN mtms_scores s ON s.event_id = e.id
WHERE ex.tradeable = 1
  AND e.status = 'extracted'
  AND s.id IS NULL;

CREATE OR REPLACE VIEW v_mtms_pending_alerts AS
SELECT e.title, e.url, e.published_at, e.feed_tier,
       ex.budget_line, ex.signal_quality, ex.monday_plan,
       ex.stand_down, ex.reverse_tickers,
       s.ticker, s.bucket, s.score, s.label,
       s.price_move_pct, s.late_vol_mult,
       s.vwap_delta_pct, s.trade_count_mult,
       s.close_position, s.event_date,
       s.id AS score_id, e.id AS event_id
FROM mtms_scores s
JOIN mtms_events e ON e.id = s.event_id
JOIN mtms_extractions ex ON ex.event_id = e.id
LEFT JOIN mtms_alerts a ON a.score_id = s.id
WHERE a.id IS NULL
ORDER BY s.score DESC, s.created_at DESC;

CREATE OR REPLACE VIEW v_mtms_event_log AS
SELECT s.event_date, s.ticker, s.bucket, s.price_move_pct,
       s.late_vol_mult, s.vwap_delta_pct, s.trade_count_mult,
       s.score, s.label,
       COALESCE(s.monday_outcome, '[ ]') AS outcome,
       e.title
FROM mtms_scores s
JOIN mtms_events e ON e.id = s.event_id
ORDER BY s.event_date DESC, s.score DESC;

CREATE OR REPLACE VIEW v_mtms_feed_report AS
SELECT e.source, e.feed_tier,
       COUNT(*) AS total,
       SUM(CASE WHEN ex.tradeable = 1 THEN 1 ELSE 0 END) AS tradeable,
       SUM(CASE WHEN e.status = 'fetch_failed' THEN 1 ELSE 0 END) AS failed,
       MAX(e.fetched_at) AS last_fetched
FROM mtms_events e
LEFT JOIN mtms_extractions ex ON ex.event_id = e.id
WHERE e.fetched_at > (NOW() - INTERVAL '7 days')::text
GROUP BY e.source, e.feed_tier
ORDER BY total DESC;
""".strip()


def print_schema():
    """Print the Postgres DDL so user can paste into Supabase SQL editor."""
    print(MTMS_SCHEMA_SQL)

# ─────────────────────────────────────────────────────────────
# ONTOLOGY
# ─────────────────────────────────────────────────────────────

def load_ontology(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

ONTOLOGY = load_ontology(ONTOLOGY_PATH)

def compute_disruption_scores(capability_ids: list[str], ontology: dict) -> dict:
    """
    For a list of capability IDs extracted from an announcement,
    compute DisruptionScore per software_function and return ranked ticker lists.

    DisruptionScore(F) = max over c in C of
        SubstituteWeight(c,F) × AutonomyMultiplier × ReliabilityMultiplier × PricingVulnerability(F)

    Returns:
        {
          "bucket_a": [tickers most directly threatened],
          "bucket_b": [tickers likely basket sympathy],
          "reverse":  [tickers likely to benefit — COMPLEMENTS],
          "scores":   {function_id: score, ...}
        }
    """
    cap_map = {c["id"]: c for c in ontology["capabilities"]}
    func_map = {f["id"]: f for f in ontology["software_functions"]}
    am = ontology["autonomy_multipliers"]
    rm = ontology["reliability_multipliers"]
    pv = ontology["pricing_vulnerability"]

    def pricing_vuln(func: dict) -> float:
        key = f"{func['pricing']}_{func['switching_cost']}_switching"
        return pv.get(key, 0.60)

    # Compute substitute scores per function
    func_scores: dict[str, float] = {}
    func_complement_scores: dict[str, float] = {}

    for edge in ontology["edges"]:
        if edge["from"] not in capability_ids:
            continue
        cap = cap_map.get(edge["from"])
        if not cap:
            continue
        func_id = edge["to"]
        if func_id not in func_map:
            continue
        func = func_map[func_id]
        weight = edge["weight"]
        multiplier = am.get(cap["autonomy"], 0.7) * rm.get(cap["reliability"], 0.7) * pricing_vuln(func)
        score = weight * multiplier

        if edge["type"] == "SUBSTITUTES":
            func_scores[func_id] = max(func_scores.get(func_id, 0), score)
        elif edge["type"] == "COMPLEMENTS":
            func_complement_scores[func_id] = max(func_complement_scores.get(func_id, 0), score)

    # Map functions → tickers
    ticker_scores: dict[str, float] = {}
    ticker_complement_scores: dict[str, float] = {}

    for t in ontology["tickers"]:
        ticker = t["ticker"]
        val_sens = {"high": 1.2, "med": 1.0, "low": 0.8}.get(t["valuation_sensitivity"], 1.0)

        # Primary functions weighted higher
        for fid in t["primary"]:
            s = func_scores.get(fid, 0) * 1.0 * val_sens
            ticker_scores[ticker] = max(ticker_scores.get(ticker, 0), s)
            c = func_complement_scores.get(fid, 0) * 1.0
            ticker_complement_scores[ticker] = max(ticker_complement_scores.get(ticker, 0), c)

        # Secondary functions weighted lower
        for fid in t["secondary"]:
            s = func_scores.get(fid, 0) * 0.6 * val_sens
            ticker_scores[ticker] = max(ticker_scores.get(ticker, 0), s)

    # Rank and bucket
    sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_complements = sorted(ticker_complement_scores.items(), key=lambda x: x[1], reverse=True)

    bucket_a = [t for t, s in sorted_tickers if s >= 0.50][:6]
    bucket_b = [t for t, s in sorted_tickers if 0.25 <= s < 0.50][:6]
    reverse  = [t for t, s in sorted_complements if s >= 0.40][:5]

    return {
        "bucket_a": bucket_a,
        "bucket_b": bucket_b,
        "reverse":  reverse,
        "function_scores": {k: round(v, 3) for k, v in sorted(func_scores.items(), key=lambda x: x[1], reverse=True)[:10]}
    }

# ─────────────────────────────────────────────────────────────
# UNIVERSE LOOKUP — sympathy tickers from scanner_universe
# ─────────────────────────────────────────────────────────────

def lookup_sympathy_tickers(bucket_a: list[str], bucket_b: list[str]) -> list[str]:
    """
    For each bucket_a ticker, find sector peers in scanner_universe.
    Returns sympathy tickers (not already in bucket_a/bucket_b).
    """
    known = set(bucket_a) | set(bucket_b)
    if not bucket_a:
        return []

    sympathy = set()
    try:
        # Get sectors for bucket_a tickers from scanner_universe
        result = (
            supabase.table("scanner_universe")
            .select("ticker,sector")
            .in_("ticker", list(bucket_a))
            .execute()
        )
        sectors = {row["sector"] for row in (result.data or []) if row.get("sector")}

        if not sectors:
            return []

        # Get all tickers in those sectors
        for sector in sectors:
            peers = (
                supabase.table("scanner_universe")
                .select("ticker")
                .eq("sector", sector)
                .execute()
            )
            for row in (peers.data or []):
                t = row["ticker"]
                if t not in known:
                    sympathy.add(t)

    except Exception as e:
        log.warning(f"Sympathy lookup error: {e}")

    return sorted(sympathy)[:20]  # cap at 20 sympathy tickers

# ─────────────────────────────────────────────────────────────
# DEDUPLICATION — 3-layer
# ─────────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    """Strip tracking params, normalize."""
    try:
        u = urlparse(url)
        clean = urlunparse((u.scheme, u.netloc, u.path, "", "", ""))
        return clean.lower().rstrip("/")
    except Exception:
        return url.lower()

def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()

def normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def title_hash(title: str) -> str:
    return hashlib.sha256(normalize_title(title).encode()).hexdigest()

# ─────────────────────────────────────────────────────────────
# LAYER 1 — INGEST WITH 3-LAYER DEDUPE
# ─────────────────────────────────────────────────────────────

def fetch_feeds() -> int:
    new_count = 0
    for feed_cfg in FEEDS:
        feed_url = feed_cfg["url"]
        tier = feed_cfg.get("tier", "A")
        feed_name = feed_cfg.get("name", feed_url)
        entry_count = 0
        feed_status = "DEAD"
        first_title = ""
        try:
            parsed = feedparser.parse(feed_url)
            source = parsed.feed.get("title", feed_name)
            entry_count = len(parsed.entries)
            feed_status = "OK" if entry_count > 0 else "DEAD"
            first_title = parsed.entries[0].title[:200] if entry_count > 0 else ""

            # Log feed health
            supabase.table("mtms_feed_health").insert({
                "feed_name": feed_name,
                "feed_url": feed_url[:500],
                "feed_tier": tier,
                "entry_count": entry_count,
                "status": feed_status,
                "last_entry_title": first_title,
            }).execute()

            for entry in parsed.entries[:30]:
                url   = entry.get("link", "").strip()
                title = entry.get("title", "").strip()
                if not url or not title:
                    continue

                uh = url_hash(url)
                th = title_hash(title)
                published = entry.get("published", "") or entry.get("updated", "")
                fetched_at = datetime.now(timezone.utc).isoformat()

                # Title-hash dedupe within 48h (use fetched_at comparison)
                cutoff = datetime.now(timezone.utc)
                cutoff_str = cutoff.isoformat()
                existing_title = (
                    supabase.table("mtms_events")
                    .select("id")
                    .eq("title_hash", th)
                    .gte("fetched_at", _hours_ago(48))
                    .limit(1)
                    .execute()
                )
                if existing_title.data:
                    continue

                # URL-hash dedupe (upsert with on_conflict skips duplicates)
                try:
                    supabase.table("mtms_events").upsert({
                        "url": url,
                        "url_hash": uh,
                        "title": title,
                        "title_hash": th,
                        "source": source,
                        "feed_tier": tier,
                        "published_at": published,
                        "fetched_at": fetched_at,
                    }, on_conflict="url_hash").execute()
                    new_count += 1
                    log.info(f"[{tier}] Ingested: {title[:70]}")
                except Exception:
                    pass  # duplicate url_hash, skip

        except Exception as e:
            log.warning(f"Feed error [{feed_url}]: {e}")

    return new_count


def _hours_ago(hours: int) -> str:
    """Return ISO timestamp for N hours ago (for gte filters)."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


def _days_ago(days: int) -> str:
    """Return ISO timestamp for N days ago."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()

# ─────────────────────────────────────────────────────────────
# LAYER 2 — TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text(url: str) -> Optional[str]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text[:10000] if text else None
    except Exception as e:
        log.warning(f"Text extraction failed [{url[:60]}]: {e}")
    return None

def enrich_with_text():
    rows = (
        supabase.table("mtms_events")
        .select("id,url")
        .is_("full_text", "null")
        .eq("status", "new")
        .limit(20)
        .execute()
    ).data or []

    for row in rows:
        text = extract_text(row["url"])
        if text:
            supabase.table("mtms_events").update({
                "full_text": text,
                "status": "fetched",
            }).eq("id", row["id"]).execute()
        else:
            supabase.table("mtms_events").update({
                "status": "fetch_failed",
            }).eq("id", row["id"]).execute()
        time.sleep(0.5)

# ─────────────────────────────────────────────────────────────
# LAYER 3 — LLM CLASSIFICATION
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an event-driven equity analyst. Evaluate AI product announcements
for tradeable market dislocations.

Shock type classification:
- demo_marketing: new UI, vague claims, no specific workflow replaced
- feature_add: improves existing product, does not kill a budget line
- workflow_replacement: AI performs an end-to-end job previously done by humans + tools  [TRADEABLE]
- cost_curve_break: same output at radically lower cost, forces repricing of incumbents  [TRADEABLE]

For tradeable events, extract:
- capability_ids: list of capability IDs from this set: codebase_understanding,
  vuln_detection_reasoning, patch_suggestion, static_analysis, code_generation,
  code_review_automation, computer_use, tool_calling_orchestration, incident_triage,
  log_analysis, threat_detection_reasoning, long_context_retrieval,
  structured_data_extraction, voice_realtime_dialog, multimodal_understanding,
  reasoning_model, rag_knowledge_retrieval, data_pipeline_generation,
  compliance_policy_reasoning, support_ticket_resolution, identity_anomaly_detection,
  network_traffic_analysis, fine_tuning, api_integration_generation, test_generation
- signal_quality: LOW=research preview, MEDIUM=beta, HIGH=generally available
- stand_down_conditions: list of conditions under which NOT to trade

Be conservative. Most announcements are noise. Prefer fewer capability IDs."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tradeable":        {"type": "boolean"},
        "shock_type":       {"type": "string"},
        "budget_line":      {"type": "string"},
        "capability_ids":   {"type": "array", "items": {"type": "string"}},
        "signal_quality":   {"type": "string", "enum": ["LOW","MEDIUM","HIGH","NONE"]},
        "monday_plan":      {"type": "string"},
        "stand_down":       {"type": "array", "items": {"type": "string"}},
        "confidence":       {"type": "number"}
    },
    "required": ["tradeable","shock_type","budget_line","capability_ids",
                 "signal_quality","monday_plan","stand_down","confidence"]
}

def classify_with_claude(title: str, text: str) -> Optional[dict]:
    """Call Claude API to classify the event and extract capability IDs."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        content = f"TITLE: {title}\n\nARTICLE:\n{text[:7000]}"

        # Ask Claude to return structured JSON matching our schema
        full_prompt = (
            SYSTEM_PROMPT + "\n\nReturn ONLY valid JSON matching this schema:\n"
            + json.dumps(EXTRACTION_SCHEMA, indent=2)
            + "\n\nArticle to classify:\n" + content
        )

        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": full_prompt}]
        )

        raw = resp.content[0].text.strip()
        # Strip any markdown code fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    except Exception as e:
        log.error(f"Claude API error: {e}")
        return None


def _save_extraction(event_id: int, result: dict):
    """Compute buckets, sympathy tickers, and save extraction to Supabase."""
    bucket_a, bucket_b, reverse_tickers, func_scores = [], [], [], {}
    sympathy_tickers = []

    if result.get("tradeable") and result.get("capability_ids"):
        scores = compute_disruption_scores(result["capability_ids"], ONTOLOGY)
        bucket_a        = scores["bucket_a"]
        bucket_b        = scores["bucket_b"]
        reverse_tickers = scores["reverse"]
        func_scores     = scores["function_scores"]
        sympathy_tickers = lookup_sympathy_tickers(bucket_a, bucket_b)

    supabase.table("mtms_extractions").insert({
        "event_id":          event_id,
        "tradeable":         int(result.get("tradeable", False)),
        "shock_type":        result.get("shock_type"),
        "budget_line":       result.get("budget_line"),
        "capability_ids":    json.dumps(result.get("capability_ids", [])),
        "bucket_a":          json.dumps(bucket_a),
        "bucket_b":          json.dumps(bucket_b),
        "sympathy_tickers":  json.dumps(sympathy_tickers),
        "reverse_tickers":   json.dumps(reverse_tickers),
        "function_scores":   json.dumps(func_scores),
        "signal_quality":    result.get("signal_quality"),
        "monday_plan":       result.get("monday_plan"),
        "stand_down":        json.dumps(result.get("stand_down", [])),
        "confidence":        result.get("confidence", 0.0),
        "raw_json":          json.dumps(result),
    }).execute()

    supabase.table("mtms_events").update({
        "status": "extracted",
    }).eq("id", event_id).execute()

    return bucket_a, bucket_b, sympathy_tickers


def interactive_review():
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Use the pending_review view
    rows = (
        supabase.table("v_mtms_pending_review")
        .select("*")
        .limit(20)
        .execute()
    ).data or []

    if not rows:
        print("\nNo pending events to review.\n")
        return

    print(f"\n{'═'*70}")
    print(f"  MTMS Interactive Review — {len(rows)} events pending")
    print(f"  Commands: 'commit' | 'skip' | 'quit' | or type feedback to Claude")
    print(f"{'═'*70}\n")

    for row in rows:
        print(f"{'─'*70}")
        print(f"  [{row['feed_tier']}] {row['title']}")
        print(f"  Source : {row['source']}")
        print(f"  URL    : {row['url'][:65]}")
        print()

        # Initial Claude classification
        content = (
            f"TITLE: {row['title']}\n\nARTICLE:\n{(row['full_text'] or '')[:5000]}\n\n"
            "Classify this event and explain your reasoning before giving the JSON. "
            "Then output the JSON classification on the last line."
        )

        full_prompt = SYSTEM_PROMPT + "\n\n" + content
        conversation = [{"role": "user", "content": full_prompt}]

        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=conversation
        )

        claude_response = resp.content[0].text.strip()
        print(f"  Claude: {claude_response[:800]}")
        print()

        conversation.append({"role": "assistant", "content": claude_response})

        # Extract JSON from Claude's response for tentative classification
        tentative = None
        try:
            json_match = re.search(r'\{[\s\S]*\}', claude_response)
            if json_match:
                tentative = json.loads(json_match.group())
        except Exception:
            pass

        # Interactive loop
        while True:
            user_input = input("  You: ").strip()

            if user_input.lower() == "quit":
                print("\n  Exiting review.\n")
                return

            if user_input.lower() == "skip":
                supabase.table("mtms_events").update({
                    "status": "skipped",
                }).eq("id", row["id"]).execute()
                print("  Skipped.\n")
                break

            if user_input.lower() == "commit":
                if tentative:
                    bucket_a, bucket_b, sympathy = _save_extraction(row["id"], tentative)
                    print(f"  Committed. Tradeable={tentative.get('tradeable')} "
                          f"Bucket A={bucket_a} Bucket B={bucket_b}")
                    if sympathy:
                        print(f"  Sympathy: {sympathy[:10]}")
                    print()
                else:
                    print("  No valid classification to commit. Skip or give Claude more guidance.\n")
                break

            # Continue conversation with Claude
            conversation.append({"role": "user", "content": user_input})
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=conversation
            )
            claude_response = resp.content[0].text.strip()
            print(f"\n  Claude: {claude_response[:600]}\n")
            conversation.append({"role": "assistant", "content": claude_response})

            # Try to extract updated classification
            try:
                json_match = re.search(r'\{[\s\S]*\}', claude_response)
                if json_match:
                    tentative = json.loads(json_match.group())
            except Exception:
                pass

def process_events():
    # Use the pending_classification view
    rows = (
        supabase.table("v_mtms_pending_classification")
        .select("*")
        .limit(10)
        .execute()
    ).data or []

    for row in rows:
        result = classify_with_claude(row["title"], row["full_text"])
        if not result:
            supabase.table("mtms_events").update({
                "status": "llm_failed",
            }).eq("id", row["id"]).execute()
            continue

        bucket_a, bucket_b, sympathy = _save_extraction(row["id"], result)
        log.info(f"Classified event {row['id']}: tradeable={result.get('tradeable')} "
                 f"type={result.get('shock_type')} confidence={result.get('confidence'):.2f}"
                 f" sympathy={len(sympathy)}")
        time.sleep(1.0)

# ─────────────────────────────────────────────────────────────
# LAYER 4 — POLYGON SCORING
# ─────────────────────────────────────────────────────────────

def get_minute_aggs(ticker: str, date_str: str) -> Optional[list]:
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
           f"/range/1/minute/{date_str}/{date_str}")
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    try:
        r = httpx.get(url, params=params, timeout=10)
        return r.json().get("results", [])
    except Exception as e:
        log.warning(f"Polygon error {ticker}: {e}")
        return None

def score_bars(bars: list, ann_epoch_ms: int) -> dict:
    pre  = [b for b in bars if b["t"] < ann_epoch_ms]
    post = [b for b in bars if b["t"] >= ann_epoch_ms]

    if not pre or not post:
        return {"score": 0, "label": "INSUFFICIENT_DATA"}

    pre_vol_avg   = sum(b["v"]           for b in pre)  / len(pre)
    post_vol_avg  = sum(b["v"]           for b in post) / len(post)
    pre_trd_avg   = sum(b.get("n", 0)   for b in pre)  / len(pre)
    post_trd_avg  = sum(b.get("n", 0)   for b in post) / len(post)

    late_vol_mult    = post_vol_avg / pre_vol_avg  if pre_vol_avg  > 0 else 0
    trade_count_mult = post_trd_avg / pre_trd_avg  if pre_trd_avg  > 0 else 0

    vwap_t0         = post[0]["vw"]
    vwap_close      = post[-1]["vw"]
    vwap_delta_pct  = (vwap_close - vwap_t0) / vwap_t0 * 100 if vwap_t0 > 0 else 0

    day_high        = max(b["h"] for b in bars)
    day_low         = min(b["l"] for b in bars)
    day_close       = bars[-1]["c"]
    open_price      = bars[0]["o"]
    price_move_pct  = (day_close - open_price) / open_price * 100
    close_position  = ((day_close - day_low) / (day_high - day_low)
                       if (day_high - day_low) > 0 else 0.5)

    score = 0
    if late_vol_mult    >= THRESHOLDS["late_vol_mult"]:    score += 1
    if vwap_delta_pct   <= THRESHOLDS["vwap_delta_pct"]:  score += 1
    if trade_count_mult >= THRESHOLDS["trade_count_mult"]: score += 1
    if close_position   <= THRESHOLDS["close_position"]:  score += 1  # bonus

    labels = {
        0: "PANIC_SHOCK_BOUNCE_LIKELY",
        1: "WEAK_SIGNAL",
        2: "CONTINUATION_BIASED",
        3: "HIGH_CONTINUATION_RISK",
        4: "HIGH_CONTINUATION_RISK"
    }

    return {
        "price_move_pct":   round(price_move_pct, 2),
        "late_vol_mult":    round(late_vol_mult, 2),
        "vwap_delta_pct":   round(vwap_delta_pct, 3),
        "trade_count_mult": round(trade_count_mult, 2),
        "close_position":   round(close_position, 3),
        "score":            score,
        "label":            labels.get(score, "UNKNOWN")
    }

def score_events():
    # Use the scoreable_events view, filter by confidence threshold
    rows = (
        supabase.table("v_mtms_scoreable_events")
        .select("*")
        .execute()
    ).data or []

    # Client-side filter for tier/confidence (view doesn't have the threshold param)
    rows = [r for r in rows
            if r["feed_tier"] == "A" or (r["confidence"] or 0) >= THRESHOLDS["min_confidence"]]

    for row in rows:
        bucket_a = json.loads(row["bucket_a"] or "[]")
        bucket_b = json.loads(row["bucket_b"] or "[]")
        date_str = (row["published_at"] or row["fetched_at"])[:10]

        try:
            ann_dt = datetime.fromisoformat(
                (row["published_at"] or row["fetched_at"]).replace("Z", "+00:00")
            )
            ann_epoch_ms = int(ann_dt.timestamp() * 1000)
        except Exception:
            ann_epoch_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        all_tickers = [(t, "A") for t in bucket_a] + [(t, "B") for t in bucket_b]

        for ticker, bucket in all_tickers:
            bars = get_minute_aggs(ticker, date_str)
            if not bars:
                continue
            metrics = score_bars(bars, ann_epoch_ms)
            supabase.table("mtms_scores").insert({
                "event_id":         row["id"],
                "ticker":           ticker,
                "bucket":           bucket,
                "event_date":       date_str,
                "price_move_pct":   metrics["price_move_pct"],
                "late_vol_mult":    metrics["late_vol_mult"],
                "vwap_delta_pct":   metrics["vwap_delta_pct"],
                "trade_count_mult": metrics["trade_count_mult"],
                "close_position":   metrics["close_position"],
                "score":            metrics["score"],
                "label":            metrics["label"],
            }).execute()
            log.info(f"Scored {ticker} ({bucket}): {metrics['score']}/4 {metrics['label']}")
            time.sleep(0.3)

        supabase.table("mtms_events").update({
            "status": "scored",
        }).eq("id", row["id"]).execute()

# ─────────────────────────────────────────────────────────────
# LAYER 5 — ALERTS + REPORTING
# ─────────────────────────────────────────────────────────────

def print_alerts():
    rows = (
        supabase.table("v_mtms_pending_alerts")
        .select("*")
        .gte("score", THRESHOLDS["min_score_to_alert"])
        .execute()
    ).data or []

    if not rows:
        log.info("No new alerts.")
        return

    W = 72
    print("\n" + "█" * W)
    print("  MTMS ALERT — Y2AI Research")
    print("█" * W)

    for row in rows:
        reverse    = json.loads(row["reverse_tickers"] or "[]")
        stand_down = json.loads(row["stand_down"] or "[]")
        print(f"\n{'─' * W}")
        print(f"  [{row['feed_tier']}] {row['title'][:65]}")
        print(f"  Date   : {row['event_date']}   Quality: {row['signal_quality']}")
        print(f"  Budget : {row['budget_line']}")
        print(f"  URL    : {row['url'][:65]}")
        print(f"\n  TICKER : {row['ticker']}  (Bucket {row['bucket']})")
        print(f"  SCORE  : {row['score']}/4  →  {row['label']}")
        print(f"  Move   : {row['price_move_pct']:+.2f}%   "
              f"VolMult: {row['late_vol_mult']:.1f}x   "
              f"VWAP Δ: {row['vwap_delta_pct']:+.3f}%   "
              f"Trades: {row['trade_count_mult']:.1f}x   "
              f"Close: {row['close_position']:.2f}")

        if row["label"] == "HIGH_CONTINUATION_RISK":
            print(f"\n  MONDAY : Bounce → VWAP rejection → continuation. Enter on failed rebound.")
        elif row["label"] == "CONTINUATION_BIASED":
            print(f"\n  MONDAY : Watch opening range. Fail below = continuation. Reclaim = abort.")
        else:
            print(f"\n  MONDAY : Watch VWAP reclaim + higher lows for long setup.")

        if reverse:
            print(f"  REVERSE: {', '.join(reverse)} may benefit (complements)")
        if stand_down:
            print(f"  STAND DOWN IF: {'; '.join(stand_down[:2])}")

        print(f"\n  Monday Plan: {row['monday_plan'][:120]}")

        supabase.table("mtms_alerts").insert({
            "event_id": row["event_id"],
            "score_id": row["score_id"],
            "ticker":   row["ticker"],
            "label":    row["label"],
            "sent_at":  datetime.now(timezone.utc).isoformat(),
        }).execute()

    print("\n" + "█" * W + "\n")


def print_event_log():
    rows = (
        supabase.table("v_mtms_event_log")
        .select("*")
        .execute()
    ).data or []

    print(f"\n{'─'*115}")
    print(f"{'Date':<12} {'Ticker':<8} {'B':<2} {'Move%':>6} "
          f"{'VolMult':>8} {'VWAPδ%':>8} {'Trds':>6} "
          f"{'Sc':>3} {'Label':<30} {'Monday'}")
    print(f"{'─'*115}")
    for r in rows:
        print(f"{r['event_date']:<12} {r['ticker']:<8} {r['bucket'] or '?':<2} "
              f"{(r['price_move_pct'] or 0):>+6.1f}% "
              f"{(r['late_vol_mult'] or 0):>7.1f}x "
              f"{(r['vwap_delta_pct'] or 0):>+7.3f}% "
              f"{(r['trade_count_mult'] or 0):>5.1f}x "
              f"{r['score']:>2}/4  "
              f"{r['label'] or '':<30} "
              f"{r['outcome']}")
    print(f"{'─'*115}")
    print("\nTo fill Monday outcome — run in Supabase SQL editor:")
    print("  UPDATE mtms_scores SET monday_outcome='CONTINUATION' "
          "WHERE ticker='X' AND event_date='YYYY-MM-DD';\n")


def print_feed_report():
    """7-day feed quality report with STALE detection."""
    rows = (
        supabase.table("v_mtms_feed_report")
        .select("*")
        .execute()
    ).data or []

    print(f"\n{'─'*90}")
    print("  MTMS Feed Quality Report — Last 7 Days")
    print(f"{'─'*90}")
    print(f"{'Source':<35} {'Tier':<5} {'Total':>6} {'Trade%':>8} {'Failed':>7} {'Last Seen':<22} {'Status'}")
    print(f"{'─'*90}")
    for r in rows:
        total     = r["total"] or 0
        tradeable = r["tradeable"] or 0
        pct       = (tradeable / total * 100) if total > 0 else 0
        per_day   = total / 7
        last      = str(r["last_fetched"] or "")[:19]

        # STALE if no entry in 48 hours
        stale = ""
        try:
            last_dt = datetime.fromisoformat(last)
            hours_ago = (datetime.now() - last_dt).total_seconds() / 3600
            if hours_ago > 48:
                stale = " ← STALE"
        except Exception:
            pass

        noise = " ← TUNE" if per_day > 10 and pct < 30 else ""
        flag  = stale or noise

        print(f"{r['source'][:34]:<35} {r['feed_tier']:<5} {total:>6} "
              f"{pct:>7.0f}% {r['failed']:>7} {last:<22}{flag}")

    print(f"{'─'*90}")
    print("Targets: 1–8 items/day per feed  |  ≥40% tradeable  |  Last seen < 48h\n")

# ─────────────────────────────────────────────────────────────
# GOOGLE SHEETS PUSH
# ─────────────────────────────────────────────────────────────

MTMS_SPREADSHEET_ID = "1pCvHBI-VeVTqkt4QdaLYdFavb4-NOMeGD_mXlgNJMeE"
GOOGLE_SHEETS_CREDS_FILE = os.path.join(os.path.dirname(_SCRIPT_DIR), "credentials.json")

EVENTS_HEADERS = [
    "ID", "Title", "Source", "Tier", "Status", "Published", "Fetched"
]
EXTRACTIONS_HEADERS = [
    "Event ID", "Tradeable", "Shock Type", "Budget Line", "Capability IDs",
    "Bucket A", "Bucket B", "Sympathy", "Reverse", "Signal Quality",
    "Confidence", "Monday Plan"
]
SCORES_HEADERS = [
    "Event ID", "Ticker", "Bucket", "Date", "Move%",
    "VolMult", "VWAP Delta%", "TradeMult", "Close Pos", "Score", "Label", "Monday"
]
ALERTS_HEADERS = [
    "Event ID", "Ticker", "Label", "Sent At", "Channel"
]


def _get_gspread_client():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_CREDS_FILE, scope)
    return gspread.authorize(creds)


def _ensure_worksheet(spreadsheet, name, headers):
    """Get or create a worksheet, ensure headers are in row 1."""
    import gspread
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=1000, cols=len(headers))
    # Set headers if empty
    existing = ws.row_values(1)
    if not existing:
        ws.update(range_name="A1", values=[headers], value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
    return ws


def push_to_sheets():
    """Push all MTMS data from Supabase to Google Sheets."""
    gc = _get_gspread_client()
    sh = gc.open_by_key(MTMS_SPREADSHEET_ID)
    log.info("Connected to MTMS Google Sheet")

    # ── Events tab ──
    events = (
        supabase.table("mtms_events")
        .select("id,title,source,feed_tier,status,published_at,fetched_at")
        .order("id")
        .execute()
    ).data or []

    ws = _ensure_worksheet(sh, "Events", EVENTS_HEADERS)
    if events:
        rows = [[
            r["id"],
            (r["title"] or "")[:120],
            r["source"] or "",
            r["feed_tier"] or "",
            r["status"] or "",
            (r["published_at"] or "")[:19],
            (r["fetched_at"] or "")[:19],
        ] for r in events]
        ws.clear()
        ws.update(range_name="A1", values=[EVENTS_HEADERS] + rows, value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
    log.info(f"  Events: {len(events)} rows")

    # ── Extractions tab ──
    extractions = (
        supabase.table("mtms_extractions")
        .select("event_id,tradeable,shock_type,budget_line,capability_ids,"
                "bucket_a,bucket_b,sympathy_tickers,reverse_tickers,"
                "signal_quality,confidence,monday_plan")
        .order("event_id")
        .execute()
    ).data or []

    ws = _ensure_worksheet(sh, "Extractions", EXTRACTIONS_HEADERS)
    if extractions:
        rows = [[
            r["event_id"],
            r["tradeable"],
            r["shock_type"] or "",
            (r["budget_line"] or "")[:120],
            r["capability_ids"] or "",
            r["bucket_a"] or "",
            r["bucket_b"] or "",
            r["sympathy_tickers"] or "",
            r["reverse_tickers"] or "",
            r["signal_quality"] or "",
            r["confidence"],
            (r["monday_plan"] or "")[:120],
        ] for r in extractions]
        ws.clear()
        ws.update(range_name="A1", values=[EXTRACTIONS_HEADERS] + rows, value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
    log.info(f"  Extractions: {len(extractions)} rows")

    # ── Scores tab ──
    scores = (
        supabase.table("mtms_scores")
        .select("event_id,ticker,bucket,event_date,price_move_pct,"
                "late_vol_mult,vwap_delta_pct,trade_count_mult,"
                "close_position,score,label,monday_outcome")
        .order("event_date", desc=True)
        .execute()
    ).data or []

    ws = _ensure_worksheet(sh, "Scores", SCORES_HEADERS)
    if scores:
        rows = [[
            r["event_id"],
            r["ticker"],
            r["bucket"] or "",
            r["event_date"] or "",
            r["price_move_pct"],
            r["late_vol_mult"],
            r["vwap_delta_pct"],
            r["trade_count_mult"],
            r["close_position"],
            r["score"],
            r["label"] or "",
            r["monday_outcome"] or "",
        ] for r in scores]
        ws.clear()
        ws.update(range_name="A1", values=[SCORES_HEADERS] + rows, value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
    log.info(f"  Scores: {len(scores)} rows")

    # ── Alerts tab ──
    alerts = (
        supabase.table("mtms_alerts")
        .select("event_id,ticker,label,sent_at,channel")
        .order("sent_at", desc=True)
        .execute()
    ).data or []

    ws = _ensure_worksheet(sh, "Alerts", ALERTS_HEADERS)
    if alerts:
        rows = [[
            r["event_id"],
            r["ticker"],
            r["label"] or "",
            (r["sent_at"] or "")[:19],
            r["channel"] or "",
        ] for r in alerts]
        ws.clear()
        ws.update(range_name="A1", values=[ALERTS_HEADERS] + rows, value_input_option="USER_ENTERED")
        ws.format("1:1", {"textFormat": {"bold": True}})
    log.info(f"  Alerts: {len(alerts)} rows")

    log.info("Google Sheets push complete")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("── MTMS Pipeline Pass ──")
    n = fetch_feeds();               log.info(f"  Ingested: {n} new events")
    enrich_with_text();              log.info("  Text extraction complete")
    process_events();                log.info("  LLM classification complete")
    score_events();                  log.info("  Polygon scoring complete")
    print_alerts()


def main():
    parser = argparse.ArgumentParser(description="MTMS Detection Pipeline — Y2AI Research")
    parser.add_argument("--schema",  action="store_true", help="Print Postgres DDL for Supabase")
    parser.add_argument("--run",     action="store_true", help="Single pipeline pass")
    parser.add_argument("--review",  action="store_true", help="Interactive Claude review")
    parser.add_argument("--alerts",  action="store_true", help="Print pending alerts")
    parser.add_argument("--log",     action="store_true", help="Print event log")
    parser.add_argument("--report",  action="store_true", help="Feed quality report")
    parser.add_argument("--push",    action="store_true", help="Push data to Google Sheets")
    parser.add_argument("--loop",    action="store_true", help="Run continuously")
    args = parser.parse_args()

    if args.schema:
        print_schema(); return

    if args.run:
        run_pipeline(); return

    if args.review:
        interactive_review(); return

    if args.alerts:
        print_alerts(); return

    if args.log:
        print_event_log(); return

    if args.report:
        print_feed_report(); return

    if args.push:
        push_to_sheets(); return

    if args.loop:
        log.info(f"Loop mode: every {FETCH_INTERVAL}s")
        while True:
            try: run_pipeline()
            except Exception as e: log.error(f"Pipeline error: {e}")
            time.sleep(FETCH_INTERVAL)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
