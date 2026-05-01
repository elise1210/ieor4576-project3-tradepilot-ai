import re
from datetime import date, datetime, timedelta
from typing import Optional


TODAY_TERMS = {"today", "current", "currently", "latest", "now", "right now"}


def parse_user_date(text: str) -> Optional[date]:
    """
    Parse common user-entered date formats.

    Supported examples:
    - today/current/latest/now/right now
    - yesterday
    - 2026-12-01
    - 2026.12.01
    - 2026/12/01
    - December 1, 2026
    - Dec 1 2026
    """
    value = (text or "").strip()
    lower = value.lower()

    if any(term in lower for term in TODAY_TERMS):
        return date.today()

    if "yesterday" in lower:
        return date.today() - timedelta(days=1)

    numeric_match = re.search(r"\b(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\b", value)
    if numeric_match:
        year, month, day = map(int, numeric_match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    month_name_match = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{1,2},?\s+20\d{2}\b",
        value,
        flags=re.IGNORECASE,
    )
    if month_name_match:
        month_text = month_name_match.group(0)
        for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
            try:
                return datetime.strptime(month_text, fmt).date()
            except ValueError:
                continue

    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None
