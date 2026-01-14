import requests
import json
import math

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def compute_confidence(matches):
    """
    Confidence is based on vector similarity scores.
    """
    if not matches:
        return 0.0

    scores = [r["score"] for r in matches]

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    # Weighted confidence (empirically stable)
    confidence = (0.7 * max_score + 0.3 * avg_score)

    # Clamp & convert to %
    confidence = max(0.0, min(confidence, 1.0))
    return round(confidence * 100, 2)


def ask_llm_with_context_streaming(
    user_query: str,
    matches: list,
    llm_model: str = "llama3.2:3b"
):
    # ---------------- BUILD CONTEXT ----------------
    context_blocks = []
    source_map = []

    for i, r in enumerate(matches, 1):
        text = r.get("paragraph_text", r["embedding_text"])
        source = r.get("source_name", "unknown")

        if "page" in r:
            source_info = f"{source} (page {r['page']})"
        elif "start_time" in r and "end_time" in r:
            source_info = f"{source} ({r['start_time']}s–{r['end_time']}s)"
        else:
            source_info = source

        context_blocks.append(f"[{i}] {text}")
        source_map.append(f"[{i}] {source_info}")

    context_text = "\n\n".join(context_blocks)
    sources_text = "\n".join(source_map)

    # ---------------- PROMPT ----------------
    prompt = f"""
You are a precise academic assistant.

RULES:
- Use ONLY the reference context.
- Do NOT add external knowledge.
- If the answer is missing, say:
  "Answer not found in the provided sources."

### Reference Context:
{context_text}

### User Question:
{user_query}

### Answer:
"""

    # ---------------- CONFIDENCE ----------------
    confidence = compute_confidence(matches)

    # ---------------- STREAM REQUEST ----------------
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        },
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    print("\n🧠 Answer (streaming):\n")

    full_answer = ""

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        token = data.get("response", "")
        print(token, end="", flush=True)
        full_answer += token

        if data.get("done"):
            break

    # ---------------- SOURCES ----------------
    print("\n\nSources:")
    for s in source_map:
        print("-", s)

    print(f"\n📊 Confidence Score: {confidence}%")

    return {
        "answer": full_answer.strip(),
        "confidence": confidence,
        "sources": source_map
    }
