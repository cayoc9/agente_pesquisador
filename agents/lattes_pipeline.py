"""
Pipeline Lattes: extracao de artigos em PDF, deduplicacao, enriquecimento e download.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import random
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from rapidfuzz import fuzz


CATALOGO_LATTES_FIELDS = [
    "record_id",
    "source_profiles",
    "publication_type",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "dedupe_key",
    "candidate_urls",
    "pdf_url",
    "local_pdf",
    "download_status",
    "download_error",
]


SECTION_HEADERS = {
    "periodico": "artigos completos publicados em periodicos",
    "anais": "trabalhos completos publicados em anais de congressos",
}

SECTION_STOP_MARKERS = {
    "livros publicados/organizados ou edicoes",
    "capitulos de livros publicados",
    "textos em jornais de noticias/revistas",
    "outras producoes bibliograficas",
    "producao tecnica",
    "demais tipos de producao tecnica",
    "orientacoes e supervisoes concluidas",
    "demais tipos de producao artistica/cultural",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


@dataclass
class LattesArticle:
    source_profile: str
    source_cv_pdf: str
    publication_type: str  # periodico | anais
    raw_text: str
    title: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    doi: str = ""


@dataclass
class LattesRecord:
    publication_type: str
    title: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    doi: str = ""
    dedupe_key: str = ""
    source_profiles: set[str] = field(default_factory=set)
    source_cv_pdfs: set[str] = field(default_factory=set)
    candidate_urls: List[str] = field(default_factory=list)
    pdf_url: str = ""
    local_pdf: str = ""
    download_status: str = "not_found"
    download_error: str = ""
    title_norm: str = ""
    skip_download: bool = False


@dataclass
class DownloadAttemptResult:
    success: bool = False
    pdf_url: str = ""
    local_pdf: str = ""
    error: str = ""
    paywalled: bool = False


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_title(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_doi(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = value.rstrip(".,;)")
    return value.lower()


def extract_doi(value: str) -> str:
    match = DOI_RE.search(value or "")
    if not match:
        return ""
    return normalize_doi(match.group(0))


def slugify(value: str, max_len: int = 80) -> str:
    base = normalize_title(value)
    base = base.replace(" ", "-")
    return (base[:max_len].strip("-") or "paper")


def infer_profile_label(pdf_path: Path) -> str:
    stem = normalize_title(pdf_path.stem)
    if "eduardo" in stem and "cerqueira" in stem:
        return "eduardo_cerqueira"
    if "denis" in stem and ("rosario" in stem or "rosario" in normalize_text(pdf_path.stem)):
        return "denis_rosario"
    return stem.replace(" ", "_") or "perfil_lattes"


def _run_pdftotext(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Comando `pdftotext` nao encontrado. Instale o pacote poppler-utils no sistema."
        ) from exc
    return result.stdout.decode("utf-8", errors="ignore")


def _find_section_bounds(lines: Sequence[str], header: str) -> Tuple[int, int]:
    start = -1
    normalized_lines = [normalize_text(line) for line in lines]
    for i, line in enumerate(normalized_lines):
        if line == header:
            start = i + 1
            break
    if start < 0:
        return -1, -1

    end = len(lines)
    for j in range(start, len(lines)):
        current = normalized_lines[j]
        if current in SECTION_STOP_MARKERS:
            end = j
            break
        if current in SECTION_HEADERS.values() and current != header:
            end = j
            break
    return start, end


def _is_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return True
    low = normalize_text(text)

    if "buscatextual.cnpq.br" in low:
        return True
    if "curriculo do sistema de curriculos lattes" in low:
        return True
    if re.fullmatch(r"\d+\s+of\s+\d+", low):
        return True
    if re.fullmatch(r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}", text):
        return True
    if low in {"ordenar por", "ordem cronologica", "citacoes:", "producao bibliografica"}:
        return True
    if low.startswith("total de trabalhos:"):
        return True
    if re.fullmatch(r"\d+\s*\|\s*\d+", low):
        return True
    return False


def _split_numbered_entries(lines: Sequence[str]) -> List[str]:
    entries: List[str] = []
    current: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if _is_noise_line(line):
            continue

        match = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match:
            if current:
                entries.append(" ".join(current).strip())
                current = []
            remainder = match.group(2).strip()
            if remainder:
                current.append(remainder)
            continue
        current.append(line)

    if current:
        entries.append(" ".join(current).strip())
    return [entry for entry in entries if entry]


def _extract_year(text: str) -> str:
    years = YEAR_RE.findall(text or "")
    return years[-1] if years else ""


def _clean_fragment(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\bCita[cç][oõ]es:\s*.*$", "", text, flags=re.IGNORECASE)
    return text.strip(" .;")


def _extract_venue(rest: str) -> str:
    rest = _clean_fragment(rest)
    parts = re.split(r"\s*,\s*(?:v\.|n\.|p\.|pp\.)", rest, maxsplit=1, flags=re.IGNORECASE)
    venue = parts[0].strip(" .;,")
    if not venue:
        fallback = re.split(r"\s+(?:19|20)\d{2}\b", rest, maxsplit=1)
        venue = fallback[0].strip(" .;,") if fallback else ""
    return venue


def _parse_entry(entry_text: str, publication_type: str, source_profile: str, source_cv_pdf: str) -> LattesArticle:
    compact = _clean_fragment(entry_text)
    doi = extract_doi(compact)
    year = _extract_year(compact)

    authors = ""
    title = compact
    venue = ""

    full_match = re.match(r"^(?P<authors>.+?)\s+\.\s+(?P<title>.+?)\.\s+(?P<rest>.+)$", compact)
    if full_match:
        authors = _clean_fragment(full_match.group("authors"))
        title = _clean_fragment(full_match.group("title"))
        venue = _extract_venue(full_match.group("rest"))
    else:
        parts = re.split(r"\s+\.\s+", compact)
        if len(parts) >= 2:
            authors = _clean_fragment(parts[0])
            title = _clean_fragment(parts[1])
            if len(parts) > 2:
                venue = _extract_venue(". ".join(parts[2:]))

    return LattesArticle(
        source_profile=source_profile,
        source_cv_pdf=source_cv_pdf,
        publication_type=publication_type,
        raw_text=entry_text,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
    )


def extract_articles_from_text(text: str, source_profile: str, source_cv_pdf: str) -> List[LattesArticle]:
    lines = text.splitlines()
    output: List[LattesArticle] = []

    for publication_type, header in SECTION_HEADERS.items():
        start, end = _find_section_bounds(lines, header)
        if start < 0:
            continue
        section_lines = lines[start:end]
        entries = _split_numbered_entries(section_lines)
        for entry in entries:
            article = _parse_entry(entry, publication_type, source_profile, source_cv_pdf)
            if article.title:
                output.append(article)

    return output


def parse_lattes_pdfs(pdf_paths: Sequence[Path]) -> List[LattesArticle]:
    articles: List[LattesArticle] = []
    for pdf_path in pdf_paths:
        text = _run_pdftotext(pdf_path)
        profile = infer_profile_label(pdf_path)
        articles.extend(extract_articles_from_text(text, profile, str(pdf_path)))
    return articles


def _merge_records(base: LattesRecord, incoming: LattesArticle) -> None:
    base.source_profiles.add(incoming.source_profile)
    base.source_cv_pdfs.add(incoming.source_cv_pdf)
    if incoming.publication_type != base.publication_type:
        existing = {p.strip() for p in base.publication_type.split(";")}
        existing.add(incoming.publication_type)
        base.publication_type = ";".join(sorted(existing))

    if incoming.title and len(incoming.title) > len(base.title):
        base.title = incoming.title
        base.title_norm = normalize_title(incoming.title)
    if incoming.authors and len(incoming.authors) > len(base.authors):
        base.authors = incoming.authors
    if incoming.venue and len(incoming.venue) > len(base.venue):
        base.venue = incoming.venue
    if not base.year and incoming.year:
        base.year = incoming.year
    if not base.doi and incoming.doi:
        base.doi = normalize_doi(incoming.doi)


def deduplicate_articles(articles: Sequence[LattesArticle], fuzzy_threshold: float = 96.0) -> List[LattesRecord]:
    records: List[LattesRecord] = []
    by_doi: Dict[str, int] = {}
    by_title_year: Dict[str, int] = {}

    for article in articles:
        article_doi = normalize_doi(article.doi)
        title_norm = normalize_title(article.title)
        title_year_key = f"title_year:{title_norm}|{article.year}" if title_norm and article.year else ""
        doi_key = f"doi:{article_doi}" if article_doi else ""

        target_idx: Optional[int] = None
        if doi_key and doi_key in by_doi:
            target_idx = by_doi[doi_key]
        elif title_year_key and title_year_key in by_title_year:
            target_idx = by_title_year[title_year_key]
        else:
            best_score = 0.0
            for idx, candidate in enumerate(records):
                if article.year and candidate.year and article.year != candidate.year:
                    continue
                if not candidate.title_norm:
                    continue
                score = fuzz.ratio(title_norm, candidate.title_norm)
                if score >= fuzzy_threshold and score > best_score:
                    best_score = score
                    target_idx = idx

        if target_idx is None:
            record = LattesRecord(
                publication_type=article.publication_type,
                title=article.title,
                authors=article.authors,
                year=article.year,
                venue=article.venue,
                doi=article_doi,
                dedupe_key=doi_key or title_year_key or f"title:{title_norm}",
                source_profiles={article.source_profile},
                source_cv_pdfs={article.source_cv_pdf},
                title_norm=title_norm,
            )
            records.append(record)
            target_idx = len(records) - 1
        else:
            _merge_records(records[target_idx], article)

        current = records[target_idx]
        if current.doi:
            current.dedupe_key = f"doi:{current.doi}"
            by_doi[current.dedupe_key] = target_idx
        elif title_year_key:
            current.dedupe_key = title_year_key
            by_title_year[title_year_key] = target_idx

        if current.title_norm and current.year:
            by_title_year[f"title_year:{current.title_norm}|{current.year}"] = target_idx

    return records


def _http_get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = 25.0,
    retries: int = 3,
) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            return response
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            sleep_for = 1.0 * (attempt + 1)
            time.sleep(sleep_for)
    if last_error:
        raise last_error
    raise RuntimeError("Falha inesperada ao executar requisicao HTTP.")


def _crossref_year(item: dict) -> Optional[int]:
    date_parts = None
    for key in ("published-print", "published-online", "created", "issued"):
        candidate = item.get(key) or {}
        date_parts = candidate.get("date-parts")
        if date_parts:
            break
    if not date_parts:
        return None
    try:
        return int(date_parts[0][0])
    except Exception:  # noqa: BLE001
        return None


def _best_crossref_match(record: LattesRecord, items: Sequence[dict]) -> Tuple[str, str]:
    query_title_norm = normalize_title(record.title)
    best_doi = ""
    best_url = ""
    best_score = 0.0
    query_year = int(record.year) if record.year.isdigit() else None

    for item in items:
        title = ((item.get("title") or [""]) or [""])[0]
        if not title:
            continue
        doi = normalize_doi(item.get("DOI", ""))
        if not doi:
            continue
        title_score = fuzz.token_set_ratio(query_title_norm, normalize_title(title))
        score = float(title_score)
        item_year = _crossref_year(item)
        if query_year and item_year:
            if item_year == query_year:
                score += 15.0
            elif abs(item_year - query_year) <= 1:
                score += 5.0
        if score > best_score:
            best_score = score
            best_doi = doi
            best_url = item.get("URL", "")

    if best_score < 70.0:
        return "", ""
    return best_doi, best_url


def _openalex_candidates(session: requests.Session, doi: str) -> List[str]:
    candidates: List[str] = []
    encoded = requests.utils.quote(doi, safe="")
    url = f"https://api.openalex.org/works/https://doi.org/{encoded}"
    try:
        response = _http_get_with_retries(session, url)
    except Exception:  # noqa: BLE001
        return candidates

    if response.status_code != 200:
        return candidates

    payload = response.json()
    open_access = payload.get("open_access") or {}
    primary = payload.get("primary_location") or {}
    locations = payload.get("locations") or []

    for value in (
        open_access.get("oa_url"),
        primary.get("pdf_url"),
        primary.get("landing_page_url"),
    ):
        if value:
            candidates.append(value)

    for location in locations[:8]:
        for value in (location.get("pdf_url"), location.get("landing_page_url")):
            if value:
                candidates.append(value)

    return candidates


def _dedupe_urls(urls: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        output.append(url)
    return output


def enrich_records_with_crossref_openalex(records: Sequence[LattesRecord]) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "agente-pesquisador-lattes/0.1"})

    for i, record in enumerate(records, 1):
        if not record.title:
            continue

        crossref_url = ""
        if not record.doi:
            query = " ".join(part for part in [record.title, record.year, record.authors] if part)
            params = {"rows": 5, "query.bibliographic": query}
            try:
                response = _http_get_with_retries(session, "https://api.crossref.org/works", params=params)
                items = (response.json().get("message") or {}).get("items") or []
                doi, crossref_url = _best_crossref_match(record, items)
                if doi:
                    record.doi = doi
                    record.dedupe_key = f"doi:{doi}"
            except Exception:  # noqa: BLE001
                pass

        candidates: List[str] = []
        if record.doi:
            candidates.append(f"https://doi.org/{record.doi}")
            candidates.extend(_openalex_candidates(session, record.doi))
        if crossref_url:
            candidates.append(crossref_url)
        record.candidate_urls = _dedupe_urls(candidates)

        if i % 20 == 0:
            time.sleep(0.2)


def load_existing_catalog(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("dedupe_key", ""): row for row in rows if row.get("dedupe_key")}


def apply_existing_catalog(records: Sequence[LattesRecord], existing: Dict[str, Dict[str, str]], force: bool) -> None:
    for record in records:
        row = existing.get(record.dedupe_key)
        if not row:
            continue

        if row.get("candidate_urls") and not record.candidate_urls:
            record.candidate_urls = _dedupe_urls(row["candidate_urls"].split("; "))

        record.pdf_url = row.get("pdf_url", record.pdf_url)
        record.local_pdf = row.get("local_pdf", record.local_pdf)
        record.download_status = row.get("download_status", record.download_status) or record.download_status
        record.download_error = row.get("download_error", record.download_error)

        if force:
            record.skip_download = False
            continue

        if record.download_status == "downloaded":
            if record.local_pdf and Path(record.local_pdf).exists():
                record.skip_download = True
            else:
                record.skip_download = False
        elif record.download_status in {"not_found", "paywalled"}:
            record.skip_download = True
        else:
            record.skip_download = False


def _is_pdf_bytes(payload: bytes) -> bool:
    return payload.startswith(b"%PDF")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_existing_hashes(output_dir: Path) -> Dict[str, Path]:
    hashes: Dict[str, Path] = {}
    if not output_dir.exists():
        return hashes
    for pdf_file in output_dir.glob("*.pdf"):
        try:
            digest = _hash_file(pdf_file)
            hashes[digest] = pdf_file
        except Exception:  # noqa: BLE001
            continue
    return hashes


def _save_pdf_bytes(
    payload: bytes,
    *,
    title: str,
    output_dir: Path,
    hash_index: Dict[str, Path],
) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    if digest in hash_index:
        return hash_index[digest]

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slugify(title)}_{digest[:10]}.pdf"
    destination = output_dir / filename
    destination.write_bytes(payload)
    hash_index[digest] = destination
    return destination


PAYWALL_HINTS = (
    "institutional access",
    "access through your institution",
    "purchase pdf",
    "purchase article",
    "subscribe to",
    "buy this article",
    "sign in to view",
    "log in to view",
    "article not available",
)


def _looks_like_paywall(text: str) -> bool:
    normalized = normalize_text(text)
    return any(hint in normalized for hint in PAYWALL_HINTS)


async def _download_pdf_via_request(context, url: str, timeout_ms: int) -> Tuple[Optional[bytes], bool, str]:
    response = await context.request.get(url, timeout=timeout_ms, fail_on_status_code=False)
    status = response.status
    if status in {401, 402, 403}:
        return None, True, f"HTTP {status}"
    if status >= 500:
        return None, False, f"HTTP {status}"
    content_type = (response.headers.get("content-type") or "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        payload = await response.body()
        if _is_pdf_bytes(payload):
            return payload, False, ""
    if "text/html" in content_type:
        try:
            html = await response.text()
            if _looks_like_paywall(html):
                return None, True, "Paywall detectado"
        except Exception:  # noqa: BLE001
            pass
    return None, False, ""


async def _extract_candidate_links(page) -> List[str]:
    links = await page.eval_on_selector_all(
        "a[href], iframe[src], embed[src], object[data]",
        """
        (elements) => elements.map((el) => {
          const href = el.href || el.src || el.data || "";
          const text = (el.innerText || el.textContent || "").trim();
          return { href, text };
        })
        """,
    )

    output: List[str] = []
    for item in links:
        href = (item.get("href") or "").strip()
        text = normalize_text(item.get("text") or "")
        if not href.startswith("http"):
            continue
        href_low = href.lower()
        if ".pdf" in href_low or "download" in href_low or "pdf" in text:
            output.append(href)
    return _dedupe_urls(output)


async def _try_download_from_candidate(
    context,
    candidate_url: str,
    record: LattesRecord,
    output_dir: Path,
    hash_index: Dict[str, Path],
    timeout_ms: int,
) -> DownloadAttemptResult:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    page = await context.new_page()
    paywalled = False
    errors: List[str] = []
    try:
        response = await page.goto(candidate_url, wait_until="domcontentloaded", timeout=timeout_ms)
        if response is not None:
            status = response.status
            content_type = (response.headers.get("content-type") or "").lower()
            if status in {401, 402, 403}:
                return DownloadAttemptResult(error=f"HTTP {status}", paywalled=True)
            if status >= 500:
                errors.append(f"HTTP {status}")
            elif "application/pdf" in content_type or page.url.lower().endswith(".pdf"):
                payload = await response.body()
                if _is_pdf_bytes(payload):
                    saved = _save_pdf_bytes(payload, title=record.title, output_dir=output_dir, hash_index=hash_index)
                    return DownloadAttemptResult(success=True, pdf_url=page.url, local_pdf=str(saved))

        try:
            html = await page.content()
            if _looks_like_paywall(html):
                paywalled = True
        except Exception:  # noqa: BLE001
            pass

        links: List[str] = []
        try:
            links = await _extract_candidate_links(page)
        except Exception:  # noqa: BLE001
            links = []

        try:
            citation_pdf = await page.query_selector("meta[name='citation_pdf_url']")
            if citation_pdf:
                citation_url = await citation_pdf.get_attribute("content")
                if citation_url:
                    links = _dedupe_urls([citation_url] + links)
        except Exception:  # noqa: BLE001
            pass

        for link in links:
            payload, is_paywalled, err = await _download_pdf_via_request(context, link, timeout_ms)
            if is_paywalled:
                paywalled = True
            if payload and _is_pdf_bytes(payload):
                saved = _save_pdf_bytes(payload, title=record.title, output_dir=output_dir, hash_index=hash_index)
                return DownloadAttemptResult(success=True, pdf_url=link, local_pdf=str(saved))
            if err:
                errors.append(err)

        click_selectors = [
            "a:has-text('PDF')",
            "a:has-text('Download PDF')",
            "a:has-text('Full Text PDF')",
            "a:has-text('Baixar PDF')",
            "button:has-text('PDF')",
        ]
        for selector in click_selectors:
            locator = page.locator(selector)
            if await locator.count() == 0:
                continue
            try:
                async with page.expect_download(timeout=4000) as download_info:
                    await locator.first.click()
                download = await download_info.value
                downloaded_path = await download.path()
                if downloaded_path:
                    payload = Path(downloaded_path).read_bytes()
                    if _is_pdf_bytes(payload):
                        saved = _save_pdf_bytes(payload, title=record.title, output_dir=output_dir, hash_index=hash_index)
                        return DownloadAttemptResult(success=True, pdf_url=page.url, local_pdf=str(saved))
            except PlaywrightTimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue

    except PlaywrightTimeoutError:
        errors.append("Timeout")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    finally:
        await page.close()

    return DownloadAttemptResult(
        success=False,
        error=" | ".join(errors[:3]),
        paywalled=paywalled,
    )


async def download_records_with_playwright(
    records: Sequence[LattesRecord],
    *,
    output_dir: Path,
    concurrency: int = 3,
    headless: bool = True,
    force: bool = False,
    timeout_ms: int = 25000,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Playwright nao esta disponivel. Instale com `pip install playwright` e rode "
            "`python -m playwright install chromium`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_index = _build_existing_hashes(output_dir)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(ignore_https_errors=True, accept_downloads=True)

        async def process(record: LattesRecord) -> None:
            if record.skip_download and not force:
                return
            if record.local_pdf and Path(record.local_pdf).exists() and not force:
                record.download_status = "downloaded"
                record.download_error = ""
                return
            if not record.candidate_urls:
                record.download_status = "not_found"
                record.download_error = "Sem candidate_urls"
                return

            async with semaphore:
                paywall_hits = 0
                collected_errors: List[str] = []
                downloaded = False

                for candidate_url in record.candidate_urls:
                    for attempt in range(3):
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                        result = await _try_download_from_candidate(
                            context,
                            candidate_url,
                            record,
                            output_dir,
                            hash_index,
                            timeout_ms,
                        )
                        if result.success:
                            record.pdf_url = result.pdf_url
                            record.local_pdf = result.local_pdf
                            record.download_status = "downloaded"
                            record.download_error = ""
                            downloaded = True
                            break

                        if result.paywalled:
                            paywall_hits += 1
                        if result.error:
                            collected_errors.append(result.error)

                        transient = result.error.startswith("HTTP 5") or "Timeout" in result.error
                        if transient and attempt < 2:
                            await asyncio.sleep(1.0 * (attempt + 1))
                            continue
                        break
                    if downloaded:
                        break

                if downloaded:
                    return
                if paywall_hits > 0:
                    record.download_status = "paywalled"
                elif collected_errors:
                    record.download_status = "error"
                else:
                    record.download_status = "not_found"
                record.download_error = " | ".join(collected_errors[:3])

        await asyncio.gather(*(process(record) for record in records))
        await context.close()
        await browser.close()


def records_to_csv_rows(records: Sequence[LattesRecord]) -> List[Dict[str, str]]:
    sorted_records = sorted(records, key=lambda item: (item.year or "", item.title_norm or normalize_title(item.title)))
    rows: List[Dict[str, str]] = []
    for idx, record in enumerate(sorted_records, 1):
        rows.append(
            {
                "record_id": f"lattes_{idx:05d}",
                "source_profiles": "; ".join(sorted(record.source_profiles)),
                "publication_type": record.publication_type,
                "title": record.title,
                "authors": record.authors,
                "year": record.year,
                "venue": record.venue,
                "doi": record.doi,
                "dedupe_key": record.dedupe_key,
                "candidate_urls": "; ".join(record.candidate_urls),
                "pdf_url": record.pdf_url,
                "local_pdf": record.local_pdf,
                "download_status": record.download_status,
                "download_error": record.download_error,
            }
        )
    return rows


def write_catalog(csv_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOGO_LATTES_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "CATALOGO_LATTES_FIELDS",
    "LattesArticle",
    "LattesRecord",
    "apply_existing_catalog",
    "deduplicate_articles",
    "download_records_with_playwright",
    "enrich_records_with_crossref_openalex",
    "extract_articles_from_text",
    "parse_lattes_pdfs",
    "records_to_csv_rows",
    "write_catalog",
]
