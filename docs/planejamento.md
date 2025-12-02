## Objetivo
Construir um agente de assistência à pesquisa científica que automatize coleta, organização e revisão de literatura, com foco inicial nos anais do SBRC (issue 1500).

## Personas e usuários-alvo
- Pesquisador que precisa mapear rapidamente artigos de um evento.
- Aluno de pós-graduação realizando survey estruturado.
- Curador técnico que precisa acompanhar tendências e identificar lacunas.

## Casos de uso iniciais
- Download e catalogação automática de artigos de um evento.
- Geração de planilha com metadados (título, autores, ano, DOI/link, palavras-chave, sessão).
- Revisão estruturada automática (resumo, método, datasets, resultados, limitações, próximos passos).
- Busca semântica por tópicos/queries no acervo baixado.
- Geração de relatório executivo e bullet points por paper.
- Monitoramento incremental (detectar novos artigos em edições futuras).

## Feature set proposto (MVP → expansão)
MVP:
- Ingestão de PDFs de conferência/evento e extração de metadados.
- Pipeline de chunking + embeddings para busca semântica.
- Agente de revisão que percorre artigos e gera ficha estruturada por paper.
- Planilha consolidada com metadados + ficha de revisão.
- CLI para rodar tarefas: `baixar`, `catalogar`, `revisar`, `buscar`.

Expansão:
- Painel web com busca e filtros (autores, tópicos, ano, score de relevância).
- Integração com Zotero/CSL para citações e exportação (BibTeX/CSL JSON).
- Alertas para novos artigos/sessões e reprocessamento incremental.
- Resumo multi-papers (sota/soat) e comparação de metodologias.
- Integração com bases abertas (CrossRef/DOAJ/ArXiv) para enriquecimento.
- Classificação automática por área/subárea (networking, segurança, SDN, IoT, 5G, edge, etc).

## Arquitetura proposta (alto nível)
- **Ingestão**: downloader (HTTP + parsing da página dos anais) → armazenamento em `docs/artigos`.
- **ETL PDF**: extração de texto (pypdf/pdfplumber), normalização, chunking.
- **Embeddings**: modelo local (e.g., `all-MiniLM-L6-v2`) via SentenceTransformers; store em Chroma/FAISS local.
- **Agentes**:
  - Agente de catalogação: lê PDFs, extrai metadados, preenche planilha base.
  - Agente de revisão: usa LangChain com prompt estruturado para gerar ficha de revisão por artigo.
  - Agente de busca: responde queries usando RAG sobre embeddings + metadados.
- **Orquestração**: scripts CLI (Typer) para rodar etapas e pipelines reentrantes.
- **Saídas**: `docs/artigos` (PDFs), `data/catalogo.csv` (metadados + revisões), `data/index/` (vetores), `reports/` (sumários executivos opcionais).

## Estrutura de diretórios sugerida
- `docs/artigos/`: PDFs baixados.
- `data/catalogo.csv`: tabela mestre com metadados e revisões.
- `data/index/`: vetores/índices semânticos.
- `agents/`: definições de chains/agents LangChain.
- `scripts/`: CLIs e utilitários (download, parse, revisão).
- `prompts/`: prompts estruturados e formatos de ficha.
- `tests/`: testes rápidos para parsers e pipelines.

## Formato da ficha de revisão (por artigo)
- Identificação: título, autores, ano, fonte, URL/DOI.
- Objetivo e problema.
- Metodologia (dados, ferramentas, modelos).
- Experimentos e resultados principais (métricas).
- Limitações/ameaças à validade.
- Contribuições e takeaways.
- Possíveis extensões/linhas futuras.
- Nível de evidência/qualidade (score 1–5).

## Métricas de sucesso
- Cobertura: % de artigos baixados e processados.
- Qualidade: avaliações manuais de 5–10 fichas (consistência e precisão).
- Tempo: duração de pipeline (download → revisão).
- Busca: precisão percebida em 5–10 queries teste.

## Próximos passos imediatos
1) Baixar PDFs do SBRC (issue 1500) e salvar em `docs/artigos`.
2) Gerar planilha inicial com metadados básicos (título, autores, link).
3) Implementar agente LangChain para preencher ficha de revisão e anexar à planilha.
4) Configurar índice de embeddings e um CLI simples para executar as etapas.
