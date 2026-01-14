import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def ask_llm_with_context(
    user_query: str,
    matches: list,
    llm_model: str = "llama3:8b"
):
    # ---------------- BUILD CONTEXT ----------------
    context_blocks = []
    source_map = []

    for i, r in enumerate(matches, 1):
        text = r.get("paragraph_text", r["embedding_text"])
        source = r.get("source_name", "unknown")

        # Handle document vs video metadata
        if "page" in r:
            source_info = f"{source} (page {r['page']})"
        elif "start_time" in r and "end_time" in r:
            source_info = f"{source} (timestamp {r['start_time']}s–{r['end_time']}s)"
        else:
            source_info = source

        context_blocks.append(f"[{i}] {text}")
        source_map.append(f"[{i}] {source_info}")

    context_text = "\n\n".join(context_blocks)
    sources_text = "\n".join(source_map)

    # ---------------- OPTIMIZED PROMPT ----------------
    prompt = f"""
You are a precise academic assistant.

RULES:
- Use ONLY the reference context provided below.
- Do NOT use prior knowledge.
- Do NOT guess or hallucinate.
- If the answer is not fully present, say:
  "Answer not found in the provided sources."
- Write a clear, structured explanation.
- At the end, include a "Sources" section.

### Reference Context:
{context_text}

### User Question:
{user_query}

### Answer Format:
<Answer in clear paragraphs>

Sources:
- file name + page number OR timestamp

### Sources Reference:
{sources_text}

### Answer:
"""

    # ---------------- CALL OLLAMA ----------------
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"].strip()
