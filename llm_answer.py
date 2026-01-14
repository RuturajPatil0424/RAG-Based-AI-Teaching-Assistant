import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def ask_local_llm(user_query, llm_model, streaming = False):
    prompt = create_prompt(user_query)

    if streaming == True:
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
    else:
        r = requests.post("http://localhost:11434/api/generate", json={
            # "model": "deepseek-r1",
            "model": llm_model,
            "prompt": prompt,
            "stream": False
        })

        response = r.json()
        return response



def create_prompt(user_query):

    prompt = f'''You are a highly intelligent, helpful, and honest AI assistant.
    
    Your task:
    - Understand the user's question carefully.
    - Provide the most accurate, clear, and complete answer possible.
    - Explain concepts simply when needed.
    - Use step-by-step reasoning for complex or technical questions.
    - Give examples or code snippets when they improve understanding.
    - If the question is ambiguous, ask for clarification.
    - If you do not know the answer or lack enough information, say so clearly instead of guessing.
    
    User question:
    {user_query}'''

    return prompt