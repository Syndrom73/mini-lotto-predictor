"""Bezpieczne pobieranie i walidacja aktualnej historii Mini Lotto."""

from __future__ import annotations

import argparse
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mini_lotto_predictor import NUMBER_COLUMNS, load_history


# Jest to maszynowy kanał CSV tej samej strony MultiPasko:
# https://www.multipasko.pl/wyniki-lotto/express-lotek
DEFAULT_URL = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto-sortowane"
SOURCE_PAGE = "https://www.multipasko.pl/wyniki-lotto/express-lotek"
WARSAW = ZoneInfo("Europe/Warsaw")


def download(
    url: str,
    destination: Path,
    require_today: bool = False,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-lotto-predictor/1.0 (+GitHub Actions)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if len(payload) < 1000:
        raise ValueError("Pobrany plik jest podejrzanie mały.")
    if b"<html" in payload[:500].lower():
        raise ValueError("Zamiast CSV pobrano stronę HTML.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    try:
        history = load_history(str(temporary))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    latest = history.iloc[-1]
    latest_date = latest["Date"].date()
    warsaw_today = datetime.now(WARSAW).date()
    if require_today and latest_date != warsaw_today:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "MultiPasko nie opublikowało jeszcze dzisiejszego wyniku Mini Lotto: "
            f"ostatnia data w źródle to {latest_date}, oczekiwano {warsaw_today}. "
            "Model nie zostanie uruchomiony na nieaktualnych danych."
        )

    os.replace(temporary, destination)

    numbers = " ".join(f"{int(latest[column]):02d}" for column in NUMBER_COLUMNS)
    print(f"Źródło: {SOURCE_PAGE}")
    print(f"Pobrano i zweryfikowano {len(history)} losowań.")
    print(f"Ostatnie: {int(latest['Numer'])}, {latest['Date'].date()}, {numbers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="wyniki-minilotto.csv")
    parser.add_argument(
        "--require-today",
        action="store_true",
        help="Przerwij, jeżeli MultiPasko nie ma jeszcze wyniku z dzisiejszej daty.",
    )
    args = parser.parse_args()
    download(args.url, Path(args.output), require_today=args.require_today)


if __name__ == "__main__":
    main()
