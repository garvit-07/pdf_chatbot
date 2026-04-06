import asyncio
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()              # ✅ Fixed: was resp.jons()
    return data.get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 180.0,
    poll_interval_s: float = 0.75,
) -> dict:
    start = time.time()
    last_status = None
    while True:
        try:
            runs = fetch_runs(event_id)
        except Exception as exc:
            st.warning(f"Polling error: {exc} — retrying…")
            time.sleep(poll_interval_s)
            continue

        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest function run ended with status: {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for run output "
                f"(last status: {last_status})"
            )
        time.sleep(poll_interval_s)


async def _send_ingest_event(pdf_path: Path) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",      # ✅ Fixed: was "rag/inngest_pdf" (typo)
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return result[0]


async def _send_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return result[0]


async def _send_summarise_event(source_id: str, top_k: int = 20) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/summarise_pdf",
            data={"source_id": source_id, "top_k": top_k},
        )
    )
    return result[0]


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files: list[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Upload & Ingest
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file", type=["pdf"], accept_multiple_files=False
    )

    if uploaded is not None:
        if uploaded.name not in st.session_state.ingested_files:
            with st.spinner(f"Ingesting **{uploaded.name}**…"):
                try:
                    path = save_uploaded_pdf(uploaded)
                    event_id = asyncio.run(_send_ingest_event(path))
                    wait_for_run_output(event_id, timeout_s=300)
                    st.session_state.ingested_files.append(uploaded.name)
                    st.success(f"✅ **{uploaded.name}** ingested!")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
        else:
            st.info(f"**{uploaded.name}** is already ingested.")

    if st.session_state.ingested_files:
        st.markdown("**Ingested files:**")
        for f in st.session_state.ingested_files:
            st.markdown(f"- 📄 `{f}`")

    st.divider()

    # Summarise button
    if st.session_state.ingested_files:
        st.subheader("📝 Summarise a PDF")
        selected = st.selectbox("Choose file to summarise", st.session_state.ingested_files)
        if st.button("Generate Summary"):
            with st.spinner("Generating summary…"):
                try:
                    event_id = asyncio.run(_send_summarise_event(selected))
                    output = wait_for_run_output(event_id, timeout_s=180)
                    summary = output.get("summary", "(No summary returned)")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"**Summary of {selected}:**\n\n{summary}"}
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Summary failed: {exc}")

    st.divider()
    st.caption("Powered by Ollama · Qdrant · Inngest")


# ──────────────────────────────────────────────────────────────────────────────
# Main — Chat interface
# ──────────────────────────────────────────────────────────────────────────────
st.title("📄 PDF RAG Chatbot")

if not st.session_state.ingested_files:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    st.caption(
        f"Chatting against: {', '.join(f'`{f}`' for f in st.session_state.ingested_files)}"
    )

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input(
    "Ask something about your PDFs…",
    disabled=not st.session_state.ingested_files,
):
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                event_id = asyncio.run(_send_query_event(question, top_k=5))
                output = wait_for_run_output(event_id, timeout_s=180)
                answer = output.get("answer", "(No answer returned)")
                sources = output.get("sources", [])
                num_ctx = output.get("num_contexts", 0)

                response_md = answer
                if sources:
                    unique_sources = list(dict.fromkeys(sources))
                    response_md += "\n\n---\n**Sources:** " + ", ".join(
                        f"`{s}`" for s in unique_sources
                    )
                    response_md += f"\n*({num_ctx} context chunks used)*"

                st.markdown(response_md)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response_md}
                )
            except Exception as exc:
                err_msg = f"⚠️ Error: {exc}"
                st.error(err_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": err_msg}
                )