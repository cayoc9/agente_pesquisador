"""
Agente de revisão de artigos usando LangChain + OpenRouter (API compatível com OpenAI) com modelo principal/fallback.

Requisitos:
    pip install -r requirements.txt
    export OPENAI_API_KEY=...

Uso básico:
    from agents.review_agent import ReviewAgent
    agent = ReviewAgent()
    review = agent.review_article(meta_dict)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class ReviewAgentConfig:
    model: str = "x-ai/grok-4.1-fast:free"
    fallback_model: str = "z-ai/glm-4.5-air:free"
    temperature: float = 0.1
    max_chars: int = 25000  # limite de texto por artigo para controle de custo
    chunk_size: int = 3500
    chunk_overlap: int = 400
    max_chunks: int = 8  # limita quantos blocos do paper serão revisados


REVIEW_FIELDS = [
    "review_resumo",
    "review_objetivo",
    "review_metodologia",
    "review_experimentos",
    "review_resultados",
    "review_limitacoes",
    "review_contribuicoes",
    "review_proximos_passos",
    "review_qualidade",
]


class ReviewAgent:
    def __init__(
        self,
        config: Optional[ReviewAgentConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.config = config or ReviewAgentConfig()
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Defina OPENROUTER_API_KEY para usar o agente de revisão.")

        default_headers = {}
        referer = os.getenv("OPENROUTER_REFERER")
        if referer:
            default_headers["HTTP-Referer"] = referer
        site = os.getenv("OPENROUTER_SITE")
        if site:
            default_headers["X-Title"] = site

        primary = os.getenv("REVIEW_MODEL_PRIMARY", self.config.model)
        fallback = os.getenv("REVIEW_MODEL_FALLBACK", self.config.fallback_model)

        self.llms = [
            ChatOpenAI(
                api_key=key,
                model=primary,
                temperature=self.config.temperature,
                base_url=self.base_url,
                default_headers=default_headers or None,
            )
        ]
        if fallback and fallback != primary:
            self.llms.append(
                ChatOpenAI(
                    api_key=key,
                    model=fallback,
                    temperature=self.config.temperature,
                    base_url=self.base_url,
                    default_headers=default_headers or None,
                )
            )
        self.note_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Você é um pesquisador técnico. Resuma objetivamente este trecho (sem opinião), "
                    "listando fatos, métodos, dados e resultados. Seja conciso (5-7 bullet points).",
                ),
                ("human", "{chunk}"),
            ]
        )
        self.review_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Você é um pesquisador sênior gerando uma ficha de revisão estruturada. "
                    "Use as anotações e o contexto fornecido. Responda em JSON válido com campos: "
                    "review_resumo, review_objetivo, review_metodologia, review_experimentos, "
                    "review_resultados, review_limitacoes, review_contribuicoes, "
                    "review_proximos_passos, review_qualidade (1-5). "
                    "Mantenha texto sintético, técnico e direto.",
                ),
                (
                    "human",
                    "Título: {title}\nAutores: {authors}\nNotas extraídas:\n{notes}\n\nGere o JSON:",
                ),
            ]
        )

    def _load_pdf_text(self, pdf_path: Path) -> str:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        text = "\n\n".join(p.page_content for p in pages)
        return text[: self.config.max_chars]

    def _chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return chunks[: self.config.max_chunks]

    def _summarize_chunks(self, chunks: List[str]) -> List[str]:
        notes: List[str] = []
        for chunk in chunks:
            resp = self._invoke_with_fallback(self.note_prompt.format_messages(chunk=chunk))
            notes.append(resp.content.strip())
        return notes

    def _final_review(self, title: str, authors: str, notes: List[str]) -> Dict[str, str]:
        notes_text = "\n\n".join(notes)
        resp = self._invoke_with_fallback(
            self.review_prompt.format_messages(title=title, authors=authors, notes=notes_text)
        )
        raw = resp.content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: devolve texto bruto se não veio JSON válido
        return {"review_resumo": raw}

    def review_article(self, meta: Dict[str, str]) -> Dict[str, str]:
        pdf_path = meta.get("local_pdf")
        if not pdf_path:
            raise ValueError("local_pdf ausente no metadado do artigo.")

        text = self._load_pdf_text(Path(pdf_path))
        chunks = self._chunk_text(text)
        notes = self._summarize_chunks(chunks)
        review = self._final_review(meta.get("title", ""), meta.get("authors", ""), notes)
        return review

    def _invoke_with_fallback(self, messages):
        last_error: Optional[Exception] = None
        for llm in self.llms:
            try:
                return llm.invoke(messages)
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("Nenhum modelo configurado para o agente de revisão.")


__all__ = ["ReviewAgent", "ReviewAgentConfig", "REVIEW_FIELDS"]
