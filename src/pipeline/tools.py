"""Function-calling / tool-use — registro de tools usadas pelo agente.

Reaproveita o LAB-001. Tool customizada: check_paper_metadata (ArXiv + Papers With Code).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable


# ============================================================================
# TODO 4 — Tool customizada do domínio: check_paper_metadata
# ============================================================================
# Consulta a API pública do ArXiv (sem chave) e retorna metadados do paper:
# título, autores, data, resumo e link para código (Papers With Code).
# Útil para perguntas do tipo: "Quem escreveu o Self-RAG?" ou
# "O paper do HyDE tem código disponível?"
# ============================================================================

_ARXIV_NS = "http://www.w3.org/2005/Atom"
_PWC_API = "https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"


def check_paper_metadata(arxiv_id: str) -> str:
    """Consulta ArXiv e Papers With Code para retornar metadados de um paper.

    Args:
        arxiv_id: ID do paper no ArXiv (ex: "2310.11511" ou "arxiv:2310.11511").

    Returns:
        String JSON com título, autores, data, resumo (primeiros 300 chars) e
        link de código, se disponível.
    """
    # Normaliza o ID — aceita formatos como "arxiv:2310.11511" ou "2310.11511"
    arxiv_id = arxiv_id.strip().removeprefix("arxiv:").removeprefix("arXiv:")

    # --- 1. Consulta API do ArXiv ---
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}&max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        return json.dumps({"error": f"Falha ao consultar ArXiv: {e}"})

    root = ET.fromstring(xml_data)
    entry = root.find(f"{{{_ARXIV_NS}}}entry")
    if entry is None:
        return json.dumps({"error": f"Paper '{arxiv_id}' não encontrado no ArXiv."})

    title = (entry.findtext(f"{{{_ARXIV_NS}}}title") or "").replace("\n", " ").strip()
    abstract = (entry.findtext(f"{{{_ARXIV_NS}}}summary") or "").replace("\n", " ").strip()
    published = (entry.findtext(f"{{{_ARXIV_NS}}}published") or "")[:10]  # só a data YYYY-MM-DD

    authors = [
        a.findtext(f"{{{_ARXIV_NS}}}name") or ""
        for a in entry.findall(f"{{{_ARXIV_NS}}}author")
    ]

    # --- 2. Consulta Papers With Code para link de código ---
    code_url: str | None = None
    try:
        pwc_url = _PWC_API.format(arxiv_id=urllib.parse.quote(arxiv_id))
        req = urllib.request.Request(pwc_url, headers={"User-Agent": "arxiv-paper-assistant/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            pwc_data = json.loads(resp.read().decode("utf-8"))
        results = pwc_data.get("results", [])
        if results:
            code_url = results[0].get("url_pdf") or results[0].get("url_abs")
            # Preferimos o repo de código, não o PDF
            repo = results[0].get("repository")
            if isinstance(repo, dict):
                code_url = repo.get("url") or code_url
    except Exception:
        pass  # Papers With Code é best-effort — não bloqueia se falhar

    result = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors[:5],           # máx 5 autores para não poluir o contexto
        "published": published,
        "abstract_snippet": abstract[:300] + ("..." if len(abstract) > 300 else ""),
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "code_url": code_url,
        "has_code": code_url is not None,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# Schema JSON para function-calling (OpenAI-compatible)
# ============================================================================

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_paper_metadata",
            "description": (
                "Consulta metadados de um paper científico pelo ID do ArXiv. "
                "Use quando o usuário perguntar sobre autores, data de publicação, "
                "resumo ou disponibilidade de código de um paper específico. "
                "Exemplo de uso: arxiv_id='2310.11511' para o Self-RAG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": (
                            "ID do paper no ArXiv. Pode ser só o número (ex: '2310.11511') "
                            "ou com prefixo (ex: 'arxiv:2310.11511')."
                        ),
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "check_paper_metadata": check_paper_metadata,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
    