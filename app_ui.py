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
    print(chunks)
    return chunks


def text_to_Embeddings(chunks):
    vector_db = BGEVectorStore(model_name="bge-m3", dim=1024, index_path="src/vector_db")
    vector_db.add_documents(chunks, batch_size=32)
    vector_db.save()

    print("✅ Vector database saved")


def import_and_process_files(path):
    split_data = text_chunking(path)
    text_to_Embeddings(split_data)


#  RUN 
def Qurey_handlder(user_query, sel_model, rag_model, answer_streaming):

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

    if rag_model == True:
        if not matches:
            print("❌ No relevant data found")
        else:
            if answer_streaming == True:
                # answer = ask_llm_with_context_streaming(
                #     user_query=user_query,
                #     matches=matches,
                #     llm_model=sel_model
                # )
                # return answer

                return "The streaming this feature is currently unavailable!\n Please continue with stream option off until we fix issue in next update!"

            else:
                answer = ask_llm_with_context(
                    user_query=user_query,
                    matches=matches,
                    llm_model=sel_model
                )
            return answer
    else:
        answer = ask_local_llm(
            user_query=user_query,
            llm_model=sel_model,
            streaming=answer_streaming
        )
        return answer
        # print("\n🧠 Final Answer:\n")
        # # print("\n📊 Confidence Score:", answer["confidence"], "%")
        # print(answer)