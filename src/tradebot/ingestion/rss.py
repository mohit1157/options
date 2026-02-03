from __future__ import annotations

import feedparser
from typing import Iterable
from datetime import datetime, timezone

from tradebot.core.events import BaseEvent

def fetch_rss_events(feeds: list[str]) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    for feed_url in feeds:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:25]:
            text = (getattr(entry, "title", "") + " " + getattr(entry, "summary", "")).strip()
            link = getattr(entry, "link", None)
            author = getattr(entry, "author", None)
            # Many RSS entries don't include reliable timestamps; use now.
            events.append(
                BaseEvent(
                    type="news",
                    source=feed_url,
                    created_at=datetime.now(timezone.utc),
                    text=text,
                    url=link,
                    author=author,
                )
            )
    return events
