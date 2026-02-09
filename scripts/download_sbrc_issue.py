"""
Baixa os artigos do SBRC (issue 1500) e gera a tabela de metadados.

Uso:
    python scripts/download_sbrc_issue.py

Saídas:
    - PDFs em docs/artigos/sbrc_<id>.pdf
    - data/catalogo.csv com metadados básicos
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
ISSUE_URL = "https://sol.sbc.org.br/index.php/sbrc/issue/view/1500"
ARTIGOS_DIR = ROOT / "docs" / "artigos" / "sbrc"
CATALOGO_PATH = ROOT / "data" / "catalogo.csv"
HEADERS = {"User-Agent": "agente-pesquisador/0.1 (+https://github.com/)"}


@dataclass
class ArticleMeta:
    article_id: str
    title: str
    authors: List[str]
    keywords: List[str]
    doi: str
    date: str
    conference: str
    firstpage: str
    lastpage: str
    issn: str
    publisher: str
    article_url: str
    pdf_url: Optional[str]
    local_pdf: Optional[str]


def ensure_dirs() -> None:
    ARTIGOS_DIR.mkdir(parents=True, exist_ok=True)
    CATALOGO_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_issue_article_ids(url: str = ISSUE_URL) -> Set[str]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    ids: Set[str] = set()
    pattern = re.compile("/article/view/(\\d+)")
    for a in soup.find_all("a", href=True):
        match = pattern.search(a["href"])
        if match:
            ids.add(match.group(1))
    return ids


def _get_meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    return tag["content"].strip() if tag and tag.get("content") else ""


def fetch_article_meta(article_id: str) -> ArticleMeta:
    article_url = f"https://sol.sbc.org.br/index.php/sbrc/article/view/{article_id}"
    resp = requests.get(article_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    title = _get_meta(soup, "citation_title") or soup.find("h1").get_text(strip=True)
    authors = [m["content"].strip() for m in soup.find_all("meta", attrs={"name": "citation_author"})]
    keywords = [m["content"].strip() for m in soup.find_all("meta", attrs={"name": "citation_keywords"})]
    return ArticleMeta(
        article_id=article_id,
        title=title,
        authors=authors,
        keywords=keywords,
        doi=_get_meta(soup, "citation_doi"),
        date=_get_meta(soup, "citation_date"),
        conference=_get_meta(soup, "citation_conference"),
        firstpage=_get_meta(soup, "citation_firstpage"),
        lastpage=_get_meta(soup, "citation_lastpage"),
        issn=_get_meta(soup, "citation_issn"),
        publisher=_get_meta(soup, "citation_publisher"),
        article_url=article_url,
        pdf_url=_get_meta(soup, "citation_pdf_url"),
        local_pdf=None,
    )


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return safe or "paper"


def download_pdf(meta: ArticleMeta) -> Optional[str]:
    if not meta.pdf_url:
        return None

    filename = f"sbrc_{meta.article_id}.pdf"
    dest = ARTIGOS_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    resp = requests.get(meta.pdf_url, headers=HEADERS, timeout=60)
    if resp.status_code != 200 or "pdf" not in resp.headers.get("Content-Type", "").lower():
        sys.stderr.write(f"[WARN] Falha no download ({resp.status_code}): {meta.pdf_url}\\n")
        return None

    dest.write_bytes(resp.content)
    return str(dest)


def write_catalog(rows: Iterable[ArticleMeta]) -> None:
    fieldnames = [
        "article_id",
        "title",
        "authors",
        "keywords",
        "doi",
        "date",
        "conference",
        "firstpage",
        "lastpage",
        "issn",
        "publisher",
        "article_url",
        "pdf_url",
        "local_pdf",
    ]
    with CATALOGO_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta in rows:
            writer.writerow(
                {
                    "article_id": meta.article_id,
                    "title": meta.title,
                    "authors": "; ".join(meta.authors),
                    "keywords": "; ".join(meta.keywords),
                    "doi": meta.doi,
                    "date": meta.date,
                    "conference": meta.conference,
                    "firstpage": meta.firstpage,
                    "lastpage": meta.lastpage,
                    "issn": meta.issn,
                    "publisher": meta.publisher,
                    "article_url": meta.article_url,
                    "pdf_url": meta.pdf_url or "",
                    "local_pdf": meta.local_pdf or "",
                }
            )


def main() -> None:
    ensure_dirs()
    print("Buscando artigos da issue...", file=sys.stderr)
    ids = sorted(fetch_issue_article_ids())
    print(f"Encontrados {len(ids)} artigos", file=sys.stderr)

    metas: List[ArticleMeta] = []
    for i, article_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] Processando artigo {article_id}", file=sys.stderr)
        meta = fetch_article_meta(article_id)
        meta.local_pdf = download_pdf(meta)
        metas.append(meta)

    write_catalog(metas)
    print(f"Tabela salva em {CATALOGO_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
