"""Streamlit UI — ArXiv RAG Paper Assistant.

Integra RAG pipeline + semantic cache + model routing + tool check_paper_metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402  <-- AGORA st está definido!

# --- Download automático do corpus (se necessário) ---
# Isso deve vir DEPOIS de importar streamlit como st
from src.pipeline.rag import build_rag_pipeline

# Define o caminho do corpus
CORPUS_DIR = _ROOT / "data" / "corpus"

# Verifica se o diretório está vazio ou não tem PDFs
if not any(CORPUS_DIR.glob("*.pdf")):
    with st.spinner("📚 Baixando corpus de papers do arXiv pela primeira vez..."):
        # Importa e executa o script de download
        import subprocess
        script_path = _ROOT / "scripts" / "download_corpus.py"
        if script_path.exists():
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Download concluído!")
                st.rerun()
            else:
                st.error(f"❌ Falha no download: {result.stderr}")
        else:
            st.warning("Script de download não encontrado. Coloque os PDFs manualmente em data/corpus/")

from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402
from src.pipeline.tools import TOOLS, run_tool_call  # noqa: E402

# ---------------------------------------------------------------- Config
st.set_page_config(
    page_title="ArXiv RAG Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 ArXiv RAG Paper Assistant")
st.caption(
    "Faça perguntas sobre os papers canônicos de RAG — "
    "o assistente responde com citação de fonte e busca metadados em tempo real."
)


# ---------------------------------------------------------------- Init lazy
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("📊 Métricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache (hits)", exact_cache.stats()["size"])
    st.metric("Semantic cache (hits)", semantic_cache.stats()["size"])

    st.divider()
    st.header("📚 Corpus")
    corpus_dir = _ROOT / "data" / "corpus"
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if pdf_files:
        for f in pdf_files:
            st.markdown(f"- `{f.stem}`")
    else:
        st.warning("Nenhum PDF encontrado em data/corpus/")

    st.divider()
    st.header("🛠 Tool disponível")
    st.markdown("`check_paper_metadata(arxiv_id)`")
    st.caption("Consulta ArXiv + Papers With Code em tempo real.")

    if st.button("🗑 Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos!")

    st.divider()
    st.header("💡 Perguntas de exemplo")
    example_questions = [
        "O que é Retrieval-Augmented Generation (RAG) e quais problemas dos LLMs ele busca resolver?",
        "Quais as principais diferenças entre RAG e Fine-tuning, e em quais cenários cada abordagem é mais recomendada?",
        "Quais são os três paradigmas de RAG descritos na literatura recente e como eles se diferenciam?",
        "Quais são as quatro habilidades fundamentais avaliadas pelo benchmark RGB para RAG, e qual delas apresentou o pior desempenho nos LLMs testados?",
        "O que a métrica faithfulness mede em um pipeline RAG e como ela se relaciona com o fenômeno de alucinação?",
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state["query_input"] = q


# ---------------------------------------------------------------- Chat
query = st.text_input(
    "Sua pergunta:",
    placeholder="Ex: Como o Self-RAG decide quando recuperar documentos?",
    key="query_input",
)

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        # 1. Exact cache
        cached = exact_cache.get(query)
        if cached:
            st.success("⚡ Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        # 2. Semantic cache
        try:
            cached = semantic_cache.get(query)
        except NotImplementedError:
            cached = None

        if cached:
            st.success("🔁 Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        # 3. Routing
        try:
            decision = classify_complexity(query)
            badge = "🟢 simples" if decision.complexity == "simple" else "🔴 complexa"
            st.info(f"**Routing:** query {badge} → `{decision.model}` | {decision.reason}")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except NotImplementedError:
            st.warning("Routing não implementado. Usando modelo default.")

        # 4. RAG pipeline
        try:
            with st.spinner("Buscando no corpus e gerando resposta..."):
                result = pipeline.answer(query)
        except NotImplementedError as e:
            st.error(f"Pipeline não implementado: {e}")
            st.stop()

        # 5. Verifica se o LLM quer chamar a tool check_paper_metadata
        #    (detecção simples por palavras-chave na pergunta)
        tool_keywords = ["arxiv", "autores", "autor", "código", "code", "publicado", "quando foi"]
        wants_tool = any(kw in query.lower() for kw in tool_keywords)

        tool_result_shown = False
        if wants_tool:
            # Tenta extrair um arxiv_id da query (padrão NNNN.NNNNN)
            import re
            match = re.search(r"\b(\d{4}\.\d{4,5})\b", query)
            if match:
                arxiv_id = match.group(1)
                with st.spinner(f"Buscando metadados do paper {arxiv_id}..."):
                    tool_raw = run_tool_call(
                        "check_paper_metadata",
                        json.dumps({"arxiv_id": arxiv_id}),
                    )
                try:
                    tool_data = json.loads(tool_raw)
                    with st.expander("🔧 Tool: check_paper_metadata", expanded=True):
                        if "error" in tool_data:
                            st.error(tool_data["error"])
                        else:
                            st.markdown(f"**Título:** {tool_data.get('title', '-')}")
                            st.markdown(f"**Autores:** {', '.join(tool_data.get('authors', []))}")
                            st.markdown(f"**Publicado:** {tool_data.get('published', '-')}")
                            st.markdown(f"**ArXiv:** {tool_data.get('arxiv_url', '-')}")
                            has_code = tool_data.get("has_code", False)
                            code_url = tool_data.get("code_url")
                            if has_code and code_url:
                                st.markdown(f"**Código:** [{code_url}]({code_url})")
                            else:
                                st.markdown("**Código:** não encontrado no Papers With Code")
                            st.markdown(f"**Resumo:** {tool_data.get('abstract_snippet', '-')}")
                    tool_result_shown = True
                    log_event("tool_called", trace_id=trace_id, tool="check_paper_metadata", arxiv_id=arxiv_id)
                except json.JSONDecodeError:
                    st.warning("Tool retornou resposta inesperada.")

        # 6. Exibe resposta do RAG
        st.markdown("### 💬 Resposta")
        st.write(result["answer"])

        if result.get("sources"):
            with st.expander("📎 Fontes citadas"):
                for source, page in sorted(result["sources"]):
                    st.write(f"- `{source}` — página {page}")

        # 7. Cacheia para próximas queries
        exact_cache.put(query, result["answer"])
        try:
            semantic_cache.put(query, result["answer"])
        except Exception:
            pass

        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))

st.divider()
st.caption(
    "**ArXiv RAG Paper Assistant** · Corpus: 12 papers canônicos de RAG · "
    "Stack: ChromaDB · Gemini · Streamlit · Tool: ArXiv API + Papers With Code"
)
