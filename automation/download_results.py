"""Bezpieczne pobieranie i walidacja aktualnej historii Mini Lotto."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from mini_lotto_predictor import NUMBER_COLUMNS, load_history


DEFAULT_URL = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto-sortowane"


def download(url: str, destination: Path) -> None:
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
    os.replace(temporary, destination)

    latest = history.iloc[-1]
    numbers = " ".join(f"{int(latest[column]):02d}" for column in NUMBER_COLUMNS)
    print(f"Pobrano i zweryfikowano {len(history)} losowań.")
    print(f"Ostatnie: {int(latest['Numer'])}, {latest['Date'].date()}, {numbers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="wyniki-minilotto.csv")
    args = parser.parse_args()
    download(args.url, Path(args.output))


if __name__ == "__main__":
    main()
