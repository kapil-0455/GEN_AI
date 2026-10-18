from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# value	    Kya hoga?
# 1.0	Relevance par zyada focus
# 0.6	Relevance ko zyada, diversity ko bhi importance
# 0.0	Diversity par zyada focus
retriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k" : 3, "lambda_mult" : 0.6}
)

results = retriver.invoke("What is LangChain?")

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)