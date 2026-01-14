
import subprocess
from pathlib import Path
from media_audio_extractor import name_list, format_converter
from text_chunker import WhisperTranscriber
from vector_database import BGEVectorStore
from rag_answer import ask_llm_with_context
from rag_answer_streaming import ask_llm_with_context_streaming
from llm_answer import ask_local_llm



VIDEO_DIR = Path("src/video")
AUDIO_DIR = Path("src/audio")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

db = BGEVectorStore(index_path="src/vector_db")



def get_local_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")[1:]
        return [line.split()[0] for line in lines]
    except Exception:
        return []

def video_to_mp3():
    available_files = [f for f in VIDEO_DIR.iterdir() if f.is_file()]
    file_map = name_list(available_files)
    format_converter(file_map, VIDEO_DIR, AUDIO_DIR, "mp3")


def text_chunking(path):
    transcriber = WhisperTranscriber()
    chunks = transcriber.ingest(path)
    print(len(chunks))
    return chunks


def text_to_Embeddings(chunks):
    vector_db = BGEVectorStore(model_name="bge-m3", dim=1024, index_path="src/vector_db")
    vector_db.add_documents(chunks, batch_size=32)
    vector_db.save()

    print("Vector database saved")

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

def import_and_process_files(path):
    split_data = text_chunking(path)
    text_to_Embeddings(split_data)

def cmd_ui():
    print("=" * 50)
    print("Welcome to the 🧠 VectorMind you currently in AI Data & Chat Interface")
    print("=" * 50)

    while True:
        user_input = input(
            "\nChoose an option:\n"
            "  - Type 'upload' to insert new data\n"
            "  - Type 'chat' to start chat mode\n"
            "  - Type 'exit' to quit\n"
            "> "
        ).strip().lower()

        if user_input == "upload":
            dir_path = input("Enter directory path or file path:")
            print("⚠️ Processing started. Do not stop the script...\n data processing will take time!")
            import_and_process_files(dir_path)

        elif user_input == "chat":
            answer_streaming = False
            sel_model = "llama3:8b"
            print("\n💬 You are now in Chat Mode.")
            print("Type 'exit' to return to main menu.\n")

            model_list = get_local_models()
            print(f"Available LLM Model : {model_list}")
            print(f"Default Model : {sel_model}")
            x = input("enter model name if you want go with default then type 'default' else type model name:")
            y = input("Do you want turn on rag mode? (y/n)")
            if x.lower() != "default":
                sel_model = x

            if y.lower() == "y" or y.lower() == "yes":
                rag_model = True

            else:
                rag_model = False

            #  RUN
            while True:

                user_query = input("Enter query: ")

                if user_query.lower() == "exit":
                    print("↩️ Exiting Chat Mode...")
                    break

                if not user_query:
                    print("⚠️ Please enter a valid message.")
                    continue

                matches = db.search_vector_db(
                    query=user_query,
                    top_k=5,
                    score_threshold=0.35
                )

                # Remove useless short chunks
                matches = [
                    r for r in matches
                    if len(r["embedding_text"].split()) >= 6
                ]

                print(rag_model)
                if rag_model == True:
                    if not matches:
                        print("No relevant data found")
                    else:
                        if answer_streaming == True:
                            answer = ask_llm_with_context_streaming(
                                user_query=user_query,
                                matches=matches,
                                llm_model=sel_model
                            )

                        else:
                            answer = ask_llm_with_context(
                                user_query=user_query,
                                matches=matches,
                                llm_model=sel_model
                            )
                        print("\nFinal Answer:\n")
                        print(answer)
                        # print("\nConfidence Score:", answer["confidence"], "%")
                else:
                    answer = ask_local_llm(
                        user_query=user_query,
                        llm_model=sel_model,
                        streaming=answer_streaming
                    )

                    print("\nFinal Answer:\n")
                    print(answer['response'])

if __name__ == "__main__":
    cmd_ui()