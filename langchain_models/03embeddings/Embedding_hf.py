from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "My name is kapil"

vector = embeddings.embed_query(text)

print(str(vector))