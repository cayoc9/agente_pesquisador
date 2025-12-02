"""
Roda o agente de revisão em lote e atualiza data/catalogo.csv.

Uso:
    python scripts/run_reviews.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

# Garante que o diretório raiz esteja no PYTHONPATH quando o script roda via `python scripts/run_reviews.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.review_agent import REVIEW_FIELDS, ReviewAgent  # noqa: E402

CATALOGO_PATH = ROOT / "data" / "catalogo.csv"


def load_catalogo() -> List[dict]:
    with CATALOGO_PATH.open() as f:
        return list(csv.DictReader(f))


def persist_catalogo(rows: List[dict], fieldnames: List[str]) -> None:
    with CATALOGO_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Roda revisões automáticas com LangChain/OpenAI")
    parser.add_argument("--limit", type=int, default=None, help="Limita quantidade de artigos processados")
    args = parser.parse_args()

    rows = load_catalogo()
    agent = ReviewAgent()

    existing_fields = list(rows[0].keys()) if rows else []
    fieldnames = existing_fields + [f for f in REVIEW_FIELDS if f not in existing_fields]

    processed = 0
    for row in tqdm(rows, desc="Revisando artigos"):
        if args.limit is not None and processed >= args.limit:
            break
        if row.get("review_resumo"):
            continue
        if not row.get("local_pdf"):
            continue

        review = agent.review_article(row)
        for f in REVIEW_FIELDS:
            row[f] = review.get(f, "")
        processed += 1

    persist_catalogo(rows, fieldnames)
    print(f"Processados {processed} artigos; catalogo atualizado em {CATALOGO_PATH}")


if __name__ == "__main__":
    main()
