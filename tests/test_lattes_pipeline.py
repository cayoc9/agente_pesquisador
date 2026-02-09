from __future__ import annotations

import unittest

from agents.lattes_pipeline import LattesArticle, deduplicate_articles, extract_articles_from_text


SAMPLE_TEXT = """
Produção bibliográfica

Artigos completos publicados em periódicos
Ordenar por
Ordem Cronológica

1.
SILVA, A. ; SOUZA, B. . A Useful Journal Paper. Journal of Tests , v. 10, p. 1-10, 2024.
2.
SANTOS, C. . Another Journal Study. IEEE Access , v. 12, p. 55-60, 2023.

Trabalhos completos publicados em anais de congressos
1.
SILVA, A. ; SOUZA, B. . Conference Result on Networks. Proceedings of NetConf, p. 100-109, 2024.

Livros publicados/organizados ou edições
"""


class LattesPipelineTests(unittest.TestCase):
    def test_extract_articles_from_text_parses_sections(self) -> None:
        articles = extract_articles_from_text(
            SAMPLE_TEXT,
            source_profile="perfil_teste",
            source_cv_pdf="/tmp/cv.pdf",
        )
        self.assertEqual(len(articles), 3)
        periodicos = [a for a in articles if a.publication_type == "periodico"]
        anais = [a for a in articles if a.publication_type == "anais"]
        self.assertEqual(len(periodicos), 2)
        self.assertEqual(len(anais), 1)
        self.assertTrue(all(a.title for a in articles))
        self.assertTrue(any(a.year == "2024" for a in articles))

    def test_dedupe_by_doi(self) -> None:
        first = LattesArticle(
            source_profile="eduardo_cerqueira",
            source_cv_pdf="/tmp/eduardo.pdf",
            publication_type="periodico",
            raw_text="",
            title="Paper with DOI",
            authors="A;B",
            year="2024",
            venue="Venue",
            doi="10.1000/XYZ-123",
        )
        second = LattesArticle(
            source_profile="denis_rosario",
            source_cv_pdf="/tmp/denis.pdf",
            publication_type="periodico",
            raw_text="",
            title="Paper with DOI",
            authors="A;B",
            year="2024",
            venue="Venue",
            doi="https://doi.org/10.1000/xyz-123",
        )
        records = deduplicate_articles([first, second])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].dedupe_key, "doi:10.1000/xyz-123")
        self.assertEqual(records[0].source_profiles, {"eduardo_cerqueira", "denis_rosario"})

    def test_dedupe_by_title_year_with_fuzzy(self) -> None:
        first = LattesArticle(
            source_profile="eduardo_cerqueira",
            source_cv_pdf="/tmp/eduardo.pdf",
            publication_type="anais",
            raw_text="",
            title="A Robust Client Selection Mechanism for Federated Learning Environments",
            authors="A",
            year="2024",
            venue="ABC",
            doi="",
        )
        second = LattesArticle(
            source_profile="denis_rosario",
            source_cv_pdf="/tmp/denis.pdf",
            publication_type="anais",
            raw_text="",
            title="A Robust Client Selection Mechanism for Federated Learning Environment",
            authors="B",
            year="2024",
            venue="ABC",
            doi="",
        )
        third = LattesArticle(
            source_profile="perfil_3",
            source_cv_pdf="/tmp/other.pdf",
            publication_type="anais",
            raw_text="",
            title="A Robust Client Selection Mechanism for Federated Learning Environment",
            authors="B",
            year="2023",
            venue="ABC",
            doi="",
        )
        records = deduplicate_articles([first, second, third])
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
