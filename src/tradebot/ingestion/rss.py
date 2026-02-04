from __future__ import annotations

import calendar
import feedparser
from datetime import datetime, timezone

from tradebot.core.events import BaseEvent


def _entry_time(entry) -> datetime:
    """Extract a best-effort timestamp from an RSS entry."""
    ts = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if ts:
        try:
            return datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)

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
                    created_at=_entry_time(entry),
                    text=text,
                    url=link,
                    author=author,
                )
            )
    return events
