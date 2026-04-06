import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
from datetime import datetime

from data_loader import load_and_chunk_pdf , embed_text
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc,RAGQueryResult,RAGSearchResult,RAGUpsertResult
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")


load_dotenv()

inngest_client = inngest.Inngest(
    app_id = "rag_app",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer()
) 
@inngest_client.create_function(
    fn_id= "RAG: Ingest pdf",
    trigger = inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx:inngest.Context):
    def _load(ctx : inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks , source_id=source_id)
    
    def _upsert(chunks_and_src : RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL , f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source":source_id,"text":chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payloads)
        return RAGUpsertResult(inngested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk",lambda: _load(ctx),output_type=RAGChunkAndSrc)
    inngested = await ctx.step.run("embed-and-upsert",lambda: _upsert(chunks_and_src),output_type=RAGUpsertResult)
    return inngested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger = inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx :inngest.Context):
    def _search(question:str,top_k:int=5) -> RAGSearchResult:
        query_vec = embed_text([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec,top_K)
        return RAGSearchResult(contexts=found["contexts"],sources=found["sources"])
        
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k",5))

    found = await ctx.step.run("embed-and-search",lambda: _search(question,top_k),output_type=RAGSearchResult)
    context_block = "\n\n".join(f-{c} for c in found.contexts)
    user_block = (
        "use the following context to anser the question.\n\n"
        f"Contexts:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    adapter=ai.openai.Adapter(
        auth_key=os.getenv("OPEN_API_KEY"),
        model= "gpt-40-mini"
    )
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body = {}
    )

app=FastAPI()

inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf])
