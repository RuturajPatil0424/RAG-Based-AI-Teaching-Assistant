from flask import Flask, render_template, request, jsonify
import subprocess
import jinja2
from app_ui import Qurey_handlder, import_and_process_files
app = Flask(__name__)

answer_streaming = False
sel_model = "llama3:8b"
rag_model = False
GUI_AI_name = "🧠 VectorMind"
project_Version = "VectorMind V3.0"
Ai_name = f"{project_Version} 3.0 - {sel_model}"
rag_mode_lbl = "RAG Mode"
streaming_mode_lbl = "Streaming"

# ---------------- OLLAMA MODELS ----------------
def get_local_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")[1:]
        model_list = [line.split()[0] for line in lines]

        if "bge-m3:latest" in model_list:
            model_list.remove("bge-m3:latest")

        return model_list
    except Exception as e:
        print("Error fetching models:", e)
        return []



# ---------------- ROUTES ----------------
@app.route("/")
def index():
    models = get_local_models()
    return render_template("index.html", models=models, GUI_AI_name=GUI_AI_name, Ai_name=Ai_name, rag_mode_lbl=rag_mode_lbl, streaming_mode_lbl=streaming_mode_lbl)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    query = data.get("query")
    sel_model = data.get("model")
    rag_model = data.get("rag")
    answer_streaming = data.get("streaming")
    Ai_name = f"{project_Version} - {sel_model}"

    print("Query:", query)
    print("Model:", sel_model)
    print("RAG:", rag_model)
    print("Streaming:", answer_streaming)

    # TEMP RESPONSE (replace later with Qurey_handlder)
    answer = Qurey_handlder(query, sel_model, rag_model, answer_streaming)
    print("Answer:", answer)

    if rag_model == True:
        return jsonify({
            "answer": answer,
            "ai_name": Ai_name
        })
    else:
        return jsonify({
            "answer": answer['response'],
            "ai_name": Ai_name
        })


if __name__ == "__main__":
    app.run(debug=True)
