# 🧠 VectorMind – RAG Based AI Assistant

VectorMind is a **local-first, Retrieval-Augmented Generation (RAG) based AI Assistant** designed to answer user queries accurately by combining **Large Language Models (LLMs)** with a **vector database** built from custom knowledge sources such as documents, websites, and videos.

Unlike traditional chatbots that rely only on pretrained knowledge, VectorMind retrieves **contextually relevant information** from your own data before generating responses—making answers **more accurate, explainable, and domain-specific**.

---

## 🚀 Key Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Local LLM support (Ollama)
- 📚 Vector database using embeddings
- ⚡ Streaming responses (token-by-token)
- 🖥️ Custom GUI (desktop-based)
- 🧩 Modular & scalable architecture
- 🔒 Offline & privacy-friendly
- 🎥 Video-to-text knowledge ingestion
- 📄 Document chunking & semantic search

---

## ❓ Problem Statement

### Traditional AI Assistants
- Hallucinate answers
- Cannot use custom or private data
- Depend heavily on cloud APIs
- Provide generic or outdated responses

### VectorMind Solves
- ❌ Hallucinations  
- ❌ Lack of domain-specific knowledge  
- ❌ Cloud dependency  
- ❌ Poor grounding of answers  

By **retrieving relevant content from a vector store before generation**, VectorMind ensures **fact-based, contextual, and reliable answers**.

---

## 🧩 System Architecture (High Level)

User Query<br>
↓<br>
Query Embedding<br>
↓<br>
Vector Store Search (FAISS)<br>
↓<br>
Relevant Context Retrieval<br>
↓<br>
Prompt Construction<br>
↓<br>
Local LLM (Ollama)<br>
↓<br>
Streaming Response to GUI


---
## 🗂️ Project Structure

VectorMind/<br>
│
├── gui.py<br>
│ └─ Handles user interface (chat, streaming, theme, actions)<br>
│
├── chunking.py<br>
│ └─ Splits documents into semantic chunks<br>
│
├── vector_store.py<br>
│ └─ Creates & manages FAISS vector database<br>
│
├── local_llm.py<br>
│ └─ Connects to Ollama local models<br>
│
├── llm_answer.py<br>
│ └─ Standard response generation<br>
│
├── llm_answer_streaming.py<br>
│ └─ Token-by-token streaming responses<br>
│
├── video_to_mp3.py<br>
│ └─ Converts video files into audio/text<br>
│
├── requirements.txt<br>
│ └─ Project dependencies<br>
│
└── README.md<br>
---
⚙️ Technologies Used

Python

Ollama (Local LLM runtime)

FAISS (Vector similarity search)

LangChain

Sentence Transformers / BGE / MiniLM

Tkinter / CustomTkinter (GUI)

Whisper / Speech-to-Text

NLTK

FFmpeg

🧪 Installation & Setup
### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/VectorMind.git
cd VectorMind

```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

```
### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 24️⃣ Install Ollama
## Download and install Ollama from:

```bash
https://ollama.com
```
## Pull a model:

```bash
ollama pull llama3:8b
# or
ollama pull llama3.2:3b
```

### 5️⃣ Download Required NLTK Data

```bash
import nltk
nltk.download('punkt')
```

### ▶️ Running the Project
## Start Ollama Server

```bash
ollama serve
```

## Launch VectorMind
```bash
python gui.py
```
---
💬 How It Works (Step-by-Step)

- User enters a query

- Query is converted into an embedding

- FAISS retrieves top relevant chunks

- Context + query are merged into a prompt

- Local LLM generates an answer

- Tokens stream live into the GUI

- User can copy or regenerate responses

---

✅ Advantages

- 🔒 Privacy-first (no cloud calls)

- 📈 Highly accurate due to RAG

- ⚡ Fast local inference

- 🔄 Reusable vector database

- 🧠 Custom knowledge support

- 🛠️ Fully customizable

- 💻 Works offline

---

⚠️ Current Limitations

- Requires local compute (RAM/GPU)

- Initial embedding generation takes time

- Large datasets increase indexing time

- GUI currently desktop-focused

---

🔮 Future Enhancements

- 🌐 Web-based interface (Flask / FastAPI)

- 📁 Multi-file batch upload

- 🔍 Hybrid search (BM25 + Vector)

- 📊 Source citation & confidence score

- 🧠 Agent-based tool calling

- 🗃️ SQLite / PostgreSQL metadata storage

- 🔐 User authentication

- ☁️ Optional cloud fallback

- 🎙️ Voice-based interaction

- 🧩 Plugin system for tools

---

🎯 Ideal Use Cases

- AI Teaching Assistant

- Internal Knowledge Base

- Resume / Interview Prep Bot

- Research Assistant

- Company Documentation Chatbot

- Offline AI Assistant

- RAG Learning & Experimentation

---

👤 Author

- Ruturaj Patil
- B.Tech Computer Engineering
- AI • ML • RAG • Python • Systems

---

⭐ Final Note

- VectorMind is built as a production-ready learning project that demonstrates real-world RAG implementation, local LLM usage, and scalable AI system design.

- If you’re learning RAG, LLM systems, or building private AI assistants, VectorMind is a strong foundation.