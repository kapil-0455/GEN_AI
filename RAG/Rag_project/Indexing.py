
from pathlib import Path

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index"


def create_index(video_id="khp7AnOJn8g"):
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        transcript_text = " ".join(
            chunk.text for chunk in transcript
        )

    except TranscriptsDisabled:
        print("No transcript available.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript_text])

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = FAISS.from_documents(
        chunks, embeddings
    )

    vectorstore.save_local(str(INDEX_PATH))

    print(f"Indexing completed. Total chunks: {len(chunks)}")


if __name__ == "__main__":
    create_index()
