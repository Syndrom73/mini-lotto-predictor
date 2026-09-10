"""Two calendar days between successful full trainings (Warsaw dates)."""
from datetime import date


def training_due(today: date, last_success: str | None) -> bool:
    return last_success is None or (today - date.fromisoformat(last_success)).days >= 2
