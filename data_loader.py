from llama_index.readers.file import PDFReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import httpx
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


EMBED_DIM = 768

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def _check_ollama_running():
    """Raise a clear, actionable error if Ollama is not reachable."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"\n\n❌ Cannot reach Ollama at {OLLAMA_BASE_URL}\n\n"
            "Fix steps:\n"
            "  1. Download Ollama: https://ollama.com/download\n"
            f"  2. Start it:        ollama serve\n"
            f"  3. Pull the model:  ollama pull {EMBED_MODEL}\n"
            "  4. Re-run this app.\n"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned an unexpected error: {exc}") from exc


def _get_embedder() -> OllamaEmbeddings:
    _check_ollama_running()
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def load_and_chunk_pdf(path: str) -> list[str]:
    """Load a PDF from disk and split it into text chunks."""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks: list[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    if not chunks:
        raise ValueError(f"No text could be extracted from: {path}")

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Checks Ollama is running before attempting."""
    embedder = _get_embedder()
    return embedder.embed_documents(texts)
