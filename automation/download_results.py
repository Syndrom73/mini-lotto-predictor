"""Bezpieczne pobieranie i walidacja aktualnej historii Mini Lotto."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from mini_lotto_predictor import NUMBER_COLUMNS, load_history


# Jest to maszynowy kanał CSV tej samej strony MultiPasko:
# https://www.multipasko.pl/wyniki-lotto/express-lotek
DEFAULT_URL = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto-sortowane"
SOURCE_PAGE = "https://www.multipasko.pl/wyniki-lotto/express-lotek"


def expected_draw_from_state(state_path: Path) -> int | None:
    """Zwraca numer losowania, dla którego zapisano ostatnią prognozę."""
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = int(state["draw_number"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Nie można odczytać oczekiwanego losowania z {state_path}.") from exc
    if expected <= 0:
        raise RuntimeError(f"Nieprawidłowy numer losowania w {state_path}: {expected}.")
    return expected


def write_github_output(new_data: bool) -> None:
    """Udostępnia wynik kolejnym krokom GitHub Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"new_data={'true' if new_data else 'false'}\n")


def download(
    url: str,
    destination: Path,
    expected_draw_number: int | None = None,
) -> bool:
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
    latest_number = int(latest["Numer"])
    latest_date = latest["Date"].date()

    # Numer prognozowanego losowania jest stabilnym identyfikatorem. Nie używamy
    # bieżącej daty, ponieważ GitHub może uruchomić zaplanowane zadanie po północy.
    if expected_draw_number is not None and latest_number < expected_draw_number:
        temporary.unlink(missing_ok=True)
        print(
            "Nowy wynik nie jest jeszcze dostępny: "
            f"źródło kończy się na losowaniu {latest_number} z {latest_date}, "
            f"oczekiwane losowanie to {expected_draw_number}."
        )
        return False

    if expected_draw_number is not None and latest_number > expected_draw_number:
        print(
            f"Uwaga: oczekiwano losowania {expected_draw_number}, ale źródło zawiera już "
            f"losowanie {latest_number}. Analiza użyje najnowszej kompletnej historii."
        )

    os.replace(temporary, destination)

    numbers = " ".join(f"{int(latest[column]):02d}" for column in NUMBER_COLUMNS)
    print(f"Źródło: {SOURCE_PAGE}")
    print(f"Pobrano i zweryfikowano {len(history)} losowań.")
    print(f"Ostatnie: {latest_number}, {latest_date}, {numbers}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="wyniki-minilotto.csv")
    parser.add_argument("--state-path", default=".automation/last_prediction.json")
    parser.add_argument(
        "--require-predicted-draw",
        action="store_true",
        help="Uruchom analizę dopiero, gdy źródło zawiera prognozowany numer losowania.",
    )
    args = parser.parse_args()

    expected = (
        expected_draw_from_state(Path(args.state_path))
        if args.require_predicted_draw
        else None
    )
    new_data = download(args.url, Path(args.output), expected_draw_number=expected)
    write_github_output(new_data)


if __name__ == "__main__":
    main()
