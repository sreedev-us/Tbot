#!/usr/bin/env python3
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import os

FEED_URL = "https://cointelegraph.com/rss"
OUTPUT_FILE = "feed/news_feed.jsonl"

def fetch_rss():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(FEED_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Failed to fetch RSS feed: {e}")
        return None

def parse_rss(xml_content):
    events = []
    try:
        root = ET.fromstring(xml_content)
        for item in root.findall('./channel/item'):
            title = item.find('title')
            title_text = title.text if title is not None else ""
            
            description = item.find('description')
            desc_text = description.text if description is not None else ""
            
            # Basic HTML stripping for description
            desc_text = desc_text.replace('<p>', '').replace('</p>', '').strip()
            
            pub_date = item.find('pubDate')
            # Fallback to current time if pubDate is missing or unparseable
            observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            categories = item.findall('category')
            tags = [cat.text for cat in categories if cat.text]
            
            if title_text:
                events.append({
                    "source": "cointelegraph",
                    "headline": title_text,
                    "body": desc_text,
                    "observedAt": observed_at,
                    "tags": tags
                })
    except Exception as e:
        print(f"Failed to parse XML: {e}")
    return events

def main():
    print(f"Fetching news from {FEED_URL}...")
    xml_content = fetch_rss()
    if not xml_content:
        return

    events = parse_rss(xml_content)
    if not events:
        print("No events parsed.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Read existing titles to avoid duplicates (naive approach)
    existing_headlines = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_headlines.add(data.get("headline", ""))
                    except json.JSONDecodeError:
                        pass
                        
    added_count = 0
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for event in reversed(events): # Oldest first so feed grows chronologically
            if event["headline"] not in existing_headlines:
                f.write(json.dumps(event) + '\n')
                added_count += 1
                
    print(f"Scraped {len(events)} articles. Added {added_count} new events to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
