"""
News scraper module for the Sentiment Oracle.
Fetches articles from crypto RSS feeds and appends them to the feed file.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

FEED_SOURCES = [
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("coindesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
]


def _fetch_xml(url: str) -> bytes | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0 TBot/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def _parse_rss(xml_content: bytes, source: str) -> list[dict]:
    events = []
    try:
        root = ET.fromstring(xml_content)
        for item in root.findall("./channel/item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            cats = [c.text or "" for c in item.findall("category")]

            headline = (title_el.text or "").strip()
            body = (desc_el.text or "").replace("<p>", "").replace("</p>", "").strip()

            if headline:
                events.append({
                    "source": source,
                    "headline": headline,
                    "body": body,
                    "observedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "tags": cats,
                })
    except Exception:
        pass
    return events


def scrape_news(feed_file: str) -> int:
    """
    Fetch fresh articles from all RSS sources and append new ones to feed_file.
    Returns the number of newly added events.
    """
    os.makedirs(os.path.dirname(feed_file) if os.path.dirname(feed_file) else ".", exist_ok=True)

    # Load existing headlines to deduplicate
    existing: set[str] = set()
    if os.path.exists(feed_file):
        with open(feed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line).get("headline", ""))
                    except json.JSONDecodeError:
                        pass

    added = 0
    with open(feed_file, "a", encoding="utf-8") as f:
        for source, url in FEED_SOURCES:
            xml_content = _fetch_xml(url)
            if not xml_content:
                continue
            for event in reversed(_parse_rss(xml_content, source)):
                if event["headline"] not in existing:
                    f.write(json.dumps(event) + "\n")
                    existing.add(event["headline"])
                    added += 1

    return added
