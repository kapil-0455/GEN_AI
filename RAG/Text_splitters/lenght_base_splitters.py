from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Dynamic path taaki kisi bhi folder se run karne par error na aaye
current_dir = Path(__file__).parent
pdf_path = current_dir / "dl-curriculum.pdf"

# 1. Load document
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()

# 2. Text Splitter setup
splitter = CharacterTextSplitter(
    chunk_size=100,        # 100 characters ka chunk
    chunk_overlap=20,      # 20 characters overlap
    separator="\n"         # PDF ke text ke hisaab se newline separator
)

# 3. Documents ko split karein ('split_documents' use karein, 'split_text' nahi)
result = splitter.split_documents(docs)

print("Total chunks:", len(result))
print("\n--- Chunk 0 content ---\n", result[0].page_content)
