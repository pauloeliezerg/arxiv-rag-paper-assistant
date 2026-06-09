"""Model routing cheap-first com fallback.

Reaproveita o notebook 05. TODO 6: classify_complexity implementado.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# Palavras que sinalizam raciocínio profundo — manda para o modelo premium
_COMPLEX_KEYWORDS = re.compile(
    r"\b(explique|explicar|compare|comparar|analise|analisar|analisa|"
    r"projete|projetar|diferencie|diferenciar|contraste|contrastar|"
    r"por que|por quê|como funciona|qual a diferença|quais as vantagens|"
    r"vantagens e desvantagens|implica|implique|avalie|avaliar|"
    r"discuta|discutir|elabore|elaborar|detalhe|detalhar|"
    r"explain|compare|analyze|analyse|contrast|differentiate|"
    r"why|how does|what are the trade.?offs|pros and cons)\b",
    re.IGNORECASE,
)

# Palavras que sinalizam busca simples de fato — modelo barato resolve
_SIMPLE_KEYWORDS = re.compile(
    r"\b(quem|quem é|quem são|quando|qual o ano|qual a data|"
    r"quantos|lista|liste|mostre|what year|who|when|list)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Regras (em ordem de prioridade):
    1. Contém palavras de análise/comparação → complex (premium)
    2. Query curta (< 60 chars) e termina em "?" e sem palavras complexas → simple
    3. Contém palavras de busca factual simples → simple
    4. Default → simple (cheap-first: errar para o lado do barato)
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    # Regra 1 — palavras que exigem raciocínio → premium
    if _COMPLEX_KEYWORDS.search(query):
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="query contém palavra-chave de análise/comparação",
        )

    # Regra 2 — query curta terminando em "?" → simple
    stripped = query.strip()
    if len(stripped) < 60 and stripped.endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="query curta (< 60 chars) terminando em '?'",
        )

    # Regra 3 — palavras de busca factual → simple
    if _SIMPLE_KEYWORDS.search(query):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="query contém palavra-chave de busca factual",
        )

    # Regra 4 — default cheap-first
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="nenhuma palavra-chave complexa detectada — usando modelo barato",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()
