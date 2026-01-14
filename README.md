# 🧠 RAG-Based AI Teaching Assistant (Local, Private, Modern)

A **full end-to-end Retrieval-Augmented Generation (RAG) system** with a **modern ChatGPT-style UI**, built using **Flask**, **FAISS**, **Whisper**, and **local LLMs via Ollama**.

This project allows you to:
- Upload PDFs and Videos
- Convert videos to audio and transcribe them
- Store document/audio knowledge in a vector database
- Ask questions using **RAG or plain LLM mode**
- Get answers with **source citations**
- Use **streaming responses**
- Switch between **light/dark themes**
- Select **local LLM models dynamically**

All processing happens **locally** — no cloud APIs required.

---

## ✨ Features

### 🔍 Core AI Features
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Local embeddings using **bge-m3**
- ✅ FAISS vector database
- ✅ Whisper-based audio/video transcription
- ✅ Semantic search with relevance filtering
- ✅ Confidence scoring based on similarity
- ✅ Source citation (PDF page / video timestamp)

### 💬 LLM & Chat
- ✅ Local LLMs via **Ollama**
- ✅ Model selector (auto-detected)
- ✅ Streaming & non-streaming answers
- ✅ Regenerate & copy answers
- ✅ Typing animation

### 🖥️ UI / UX
- ✅ ChatGPT-style sidebar layout
- ✅ Chat bubbles (user right, AI left)
- ✅ Floating input labels
- ✅ Toggle switches (RAG / Streaming)
- ✅ File upload in sidebar
- ✅ Light / Dark mode (persistent)
- ✅ Clean, modern SaaS-style UI

### 📁 File Support
- PDFs (`.pdf`)
- Videos (`.mp4`, `.mkv`, `.avi`)
- Audio extracted automatically from videos

---

## 🧩 Project Architecture

