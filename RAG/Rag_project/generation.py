
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from retrival import get_retriever


llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


def format_docs(retrieved_docs):
    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


def build_chain():
    retriever = get_retriever()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel_chain | prompt | llm | StrOutputParser()


def answer_question(question):
    chain = build_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    question = input("Ask a question about the video: ")
    answer = answer_question(question)
    print("\nAnswer:", answer)
