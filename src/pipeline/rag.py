"""RAG pipeline — chunk, embed, index, retrieve, generate.

Reaproveita as funcoes do notebook 02. Voce vai preencher 3 TODOs aqui.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _make_client() -> tuple[OpenAI, str]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma.

        Retorna numero de chunks indexados.
        """
        # TODO 1.A — Ler todos os PDFs e extrair texto por página
        docs: list[dict] = []
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"Nenhum PDF encontrado em {self.corpus_dir}")

        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:          # pula páginas em branco
                    continue
                docs.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_num,
                })

        # TODO 1.B — Chunking recursivo: 800 chars, overlap 100
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[dict] = []
        for doc in docs:
            sub_texts = splitter.split_text(doc["text"])
            for i, sub in enumerate(sub_texts):
                chunk_id = f"{doc['source']}__p{doc['page']}__c{i}"
                chunks.append({
                    "id": chunk_id,
                    "text": sub,
                    "source": doc["source"],
                    "page": doc["page"],
                })

        # TODO 1.C — Indexar no Chroma em lotes de 100 (evita timeout de embedding)
        BATCH = 100
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            )

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        results = self.collection.query(query_texts=[query], n_results=k)

        hits: list[dict] = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": text,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "distance": dist,
            })
        return hits

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)

        # 1. Montar contexto com cabeçalho [source:page] para cada chunk
        context_parts = []
        for h in hits:
            header = f"[{h['source']}:p{h['page']}]"
            context_parts.append(f"{header}\n{h['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # 2. Montar prompt com o template do projeto
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. Chamar o LLM
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,       # respostas mais factuais
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()

        # 4. Retornar resposta + fontes únicas
        sources = list({(h["source"], h["page"]) for h in hits})
        return {"answer": answer_text, "sources": sources}


PROMPT_TEMPLATE = """Voce e um assistente tecnico especializado em pesquisa de ML/IA.
Responda APENAS com base no contexto abaixo — cite sempre a fonte no formato [arquivo:pagina].
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
