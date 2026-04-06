# 📄 PDF Chatbot using RAG (Qdrant + Ollama + Inngest + Docker)

## Overview
This project is an end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and interact with them through a chatbot.

The system processes the uploaded PDF, converts it into embeddings, stores them in a **vector database (Qdrant)**, and uses a **local LLM (Ollama)** to generate context-aware answers.

It also integrates **Inngest** for handling backend workflows and is fully containerized using **Docker**.

---

##  Key Features
- 📤 Upload PDF documents
- 🔍 Convert text into vector embeddings
- 🗄️ Store embeddings in Qdrant vector database
- 💬 Chat with your PDF (context-aware responses)
- ⚡ Backend workflows powered by Inngest
- 🐳 Fully Dockerized for easy deployment
- 🔒 Runs locally using Ollama (no external API required)

---

##  Tech Stack

### 🔹 Frontend
- Streamlit

### 🔹 Backend
- Python
- Inngest (event-driven backend workflows)

### 🔹 AI / ML
- Embeddings + RAG pipeline
- Ollama (Local LLM)

### 🔹 Database
- Qdrant (Vector Database)

### 🔹 DevOps
- Docker

---

## ⚙️ How It Works

1. **PDF Upload**
   - User uploads a PDF via the UI

2. **Text Processing**
   - PDF is parsed and split into chunks

3. **Embedding Generation**
   - Each chunk is converted into vector embeddings

4. **Vector Storage**
   - Embeddings are stored in Qdrant

5. **Query Handling**
   - User asks a question
   - Relevant chunks are retrieved from Qdrant

6. **Response Generation**
   - Retrieved context is passed to Ollama LLM
   - Model generates a contextual answer

