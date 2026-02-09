# Entrada de CVs Lattes

Coloque nesta pasta os curriculos Lattes em PDF que serao processados pelo script:

```bash
python scripts/download_lattes_articles.py
```

Regras:
- O script busca automaticamente arquivos `*.pdf` e `*.PDF` dentro de `input/lattes/`.
- Para usar outro caminho, rode com `--input-dir`.
- Arquivos PDF desta pasta sao ignorados pelo git (`.gitignore`).
