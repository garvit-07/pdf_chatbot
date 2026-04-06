import logging
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_ollama import OllamaLLM

from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult
from data_loader import embed_texts, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

# ✅ FREE local LLM via Ollama
llm = OllamaLLM(model="llama3")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# =============================================================================
# 🔹 INGEST FUNCTION
# =============================================================================
@inngest_client.create_function(
    fn_id="RAG: Ingest pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Triggered by event: rag/ingest_pdf
    Expected event data:
        pdf_path  (str)  – absolute path to the PDF file
        source_id (str)  – optional label; defaults to pdf_path
    """

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)

        # ✅ Fixed: was RAGUpsertResult(inngested=...) — double 'n' typo
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    result = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return result.model_dump()


# =============================================================================
# 🔹 QUERY FUNCTION
# =============================================================================
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """
    Triggered by event: rag/query_pdf_ai
    Expected event data:
        question (str)  – the user's question
        top_k    (int)  – number of chunks to retrieve (default 5)
    """

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )

    def _generate(question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I couldn't find any relevant information in the uploaded documents."

        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say so clearly.

Context:
{context_block}

Question: {question}

Answer concisely and accurately based on the context above."""

        return llm.invoke(prompt)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    answer = await ctx.step.run(
        "generate-answer",
        lambda: _generate(question, found.contexts),
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# =============================================================================
# 🔹 SUMMARISE FUNCTION (bonus — summarises a whole ingested PDF)
# =============================================================================
@inngest_client.create_function(
    fn_id="RAG: Summarise PDF",
    trigger=inngest.TriggerEvent(event="rag/summarise_pdf"),
)
async def rag_summarise_pdf(ctx: inngest.Context):
    """
    Triggered by event: rag/summarise_pdf
    Expected event data:
        source_id (str) – the source_id used when the PDF was ingested
        top_k     (int) – how many chunks to pull for the summary (default 20)
    """

    def _fetch_and_summarise(source_id: str, top_k: int) -> str:
        # Use a generic "summarise this document" query to pull representative chunks
        query_vec = embed_texts([f"summary overview of {source_id}"])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)

        if not found["contexts"]:
            return "No content found for this source."

        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(found["contexts"]))

        prompt = f"""Summarise the following document excerpts into a clear, concise summary.
Cover the main topics, key points, and conclusions.

Excerpts:
{context_block}

Summary:"""

        return llm.invoke(prompt)

    source_id = ctx.event.data["source_id"]
    top_k = int(ctx.event.data.get("top_k", 20))

    summary = await ctx.step.run(
        "fetch-and-summarise",
        lambda: _fetch_and_summarise(source_id, top_k),
    )

    return {"summary": summary, "source_id": source_id}


# =============================================================================
# 🔹 FASTAPI APP
# =============================================================================
app = FastAPI(title="RAG PDF Chatbot", version="1.0.0")

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai, rag_summarise_pdf],
)