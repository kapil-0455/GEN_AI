
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.retrievers import MultiQueryRetriever

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

chat_model = ChatOllama(
    model="llama3.2:1b",
)

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Create FAISS vector store
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings
)

# 1. Similarity retriever
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 2. MultiQuery retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=chat_model
)

query = "How to improve energy levels and maintain balance?"

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

print("\n========== SIMILARITY RESULTS ==========")

for i, doc in enumerate(similarity_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))

print("\n========== MULTIQUERY RESULTS ==========")

for i, doc in enumerate(multiquery_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))
