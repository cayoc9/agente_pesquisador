"""
Baixa artigos unicos (periodicos + anais) a partir de CVs Lattes em PDF.

Uso:
    python scripts/download_lattes_articles.py
    python scripts/download_lattes_articles.py --max-items 10 --headful
    python scripts/download_lattes_articles.py --input-dir input/lattes
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Garante que o diretorio raiz esteja no PYTHONPATH quando executado via python scripts/...
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.lattes_pipeline import (  # noqa: E402
    apply_existing_catalog,
    deduplicate_articles,
    download_records_with_playwright,
    enrich_records_with_crossref_openalex,
    load_existing_catalog,
    parse_lattes_pdfs,
    records_to_csv_rows,
    write_catalog,
)

DEFAULT_INPUT_DIR = Path(os.getenv("LATTES_INPUT_DIR", str(ROOT / "input" / "lattes")))
CATALOGO_LATTES_PATH = ROOT / "data" / "catalogo_lattes.csv"
LATTES_ARTIGOS_DIR = ROOT / "docs" / "artigos" / "lattes"


def _resolve_cv_paths(cli_values: List[str] | None, input_dir: Path) -> List[Path]:
    if cli_values:
        return [Path(item).expanduser().resolve() for item in cli_values]

    candidates = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    return sorted(path.resolve() for path in candidates if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Lattes: parse, dedupe, enriquecimento e download de PDFs")
    parser.add_argument(
        "--cv-pdf",
        action="append",
        dest="cv_pdfs",
        default=None,
        help="PDF do curriculo Lattes. Pode repetir o argumento para multiplos curriculos.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Diretorio de entrada com CVs Lattes em PDF (usado quando --cv-pdf nao for informado).",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Limita quantidade de itens para smoke run.")
    parser.add_argument("--concurrency", type=int, default=3, help="Quantidade de downloads em paralelo.")
    parser.add_argument("--force", action="store_true", help="Reprocessa itens mesmo que ja estejam no catalogo.")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Executa Chromium sem UI.")
    parser.add_argument("--headful", dest="headless", action="store_false", help="Executa Chromium com UI.")
    parser.set_defaults(headless=True)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    cv_paths = _resolve_cv_paths(args.cv_pdfs, input_dir)

    if args.cv_pdfs is None and not input_dir.exists():
        raise FileNotFoundError(
            f"Diretorio de entrada nao encontrado: {input_dir}. "
            "Crie a pasta e adicione os curriculos PDF, ou use --cv-pdf."
        )

    missing = [str(path) for path in cv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PDF(s) Lattes nao encontrado(s): {', '.join(missing)}")
    if not cv_paths:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {input_dir}. "
            "Adicione os curriculos na pasta input ou use --cv-pdf."
        )

    if args.cv_pdfs:
        print(f"[1/5] Extraindo artigos dos PDFs ({len(cv_paths)} curriculos via --cv-pdf)...")
    else:
        print(f"[1/5] Extraindo artigos dos PDFs em {input_dir} ({len(cv_paths)} curriculos)...")
    articles = parse_lattes_pdfs(cv_paths)
    print(f"  -> artigos extraidos (bruto): {len(articles)}")

    print("[2/5] Deduplicando artigos...")
    records = deduplicate_articles(articles)
    print(f"  -> artigos unicos: {len(records)}")

    if args.max_items is not None:
        records = records[: args.max_items]
        print(f"  -> limite aplicado: {len(records)} registro(s)")

    print("[3/5] Enriquecendo com Crossref/OpenAlex...")
    enrich_records_with_crossref_openalex(records)

    existing = load_existing_catalog(CATALOGO_LATTES_PATH)
    apply_existing_catalog(records, existing, force=args.force)

    print("[4/5] Baixando PDFs com Playwright...")
    asyncio.run(
        download_records_with_playwright(
            records,
            output_dir=LATTES_ARTIGOS_DIR,
            concurrency=args.concurrency,
            headless=args.headless,
            force=args.force,
        )
    )

    print("[5/5] Gravando catalogo...")
    rows = records_to_csv_rows(records)
    write_catalog(CATALOGO_LATTES_PATH, rows)

    downloaded = sum(1 for row in rows if row["download_status"] == "downloaded")
    paywalled = sum(1 for row in rows if row["download_status"] == "paywalled")
    not_found = sum(1 for row in rows if row["download_status"] == "not_found")
    errors = sum(1 for row in rows if row["download_status"] == "error")
    print(f"Catalogo salvo em: {CATALOGO_LATTES_PATH}")
    print(
        "Resumo -> "
        f"downloaded={downloaded}, paywalled={paywalled}, not_found={not_found}, error={errors}, total={len(rows)}"
    )


if __name__ == "__main__":
    main()
